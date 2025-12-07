import torch
import torch.nn as nn
import numpy as np
import scipy.fftpack


class SafeSplitDefense:
    def __init__(self, config):
        self.history_len = config['defense']['history_len']
        self.backbone_history = []  # Stores state_dicts
        self.threshold_ratio = config['defense']['safe_majority_threshold']
        self.device = config['system']['device']

    def update_history(self, backbone_model):
        """Adds current backbone state to history (FIFO)"""
        # Deep copy to ensure we store the snapshot
        state = {k: v.cpu().clone()
                 for k, v in backbone_model.state_dict().items()}
        self.backbone_history.append(state)
        if len(self.backbone_history) > self.history_len:
            self.backbone_history.pop(0)

    def get_latest_valid_model_index(self):
        """
        Executes the SafeSplit logic:
        1. Static Analysis (DCT)
        2. Dynamic Analysis (Rotational Distance)
        Returns: Index of the latest model in history that is part of the benign majority.
                 Returns -1 if current model is benign.
        """
        if len(self.backbone_history) < self.history_len:
            # Not enough history to judge
            return -1

        # N is number of clients in history window
        N = len(self.backbone_history)

        # --- 1. FREQUENCY ANALYSIS (Static) ---
        freq_scores = self._compute_frequency_scores(N)

        # --- 2. ROTATIONAL ANALYSIS (Dynamic) ---
        rot_scores = self._compute_rotational_scores(N)

        # --- 3. IDENTIFY MAJORITY ---
        # "Smallest N/2 + 1"
        k = int(N / 2) + 1

        # Get indices of the k smallest scores
        freq_majority_indices = np.argsort(freq_scores)[:k]
        rot_majority_indices = np.argsort(rot_scores)[:k]

        # Intersection of majorities (Strict SafeSplit) or union?
        # Paper says: "Check if current client i is in BOTH majority groups"

        # We check the models in reverse order (Latest -> Oldest)
        # The history list is [Oldest, ..., Latest]
        # Indices in history: 0 to N-1. Latest is N-1.

        for i in range(N - 1, -1, -1):
            is_freq_safe = i in freq_majority_indices
            is_rot_safe = i in rot_majority_indices

            if is_freq_safe and is_rot_safe:
                # Found the latest benign model
                if i == N - 1:
                    return -1  # Current model is safe
                else:
                    return i  # Return index of safe historical model to rollback to

        # Fallback: if no model satisfies both, return earliest (safest bet) or crash
        return 0

    def _compute_frequency_scores(self, N):
        """
        Compute Euclidean distance of Low-Freq DCT components of updates.
        S_t = DCT_low(B_t - B_{t-1})
        """
        scores = np.zeros(N)
        # Pre-compute updates and their DCTs
        dct_reps = []

        for t in range(N):
            curr = self.backbone_history[t]
            # First one has 0 diff
            prev = self.backbone_history[t-1] if t > 0 else curr

            # Flatten all params to one vector for simplicity or layer-wise
            # Paper implies whole backbone.
            diff_vec = []
            for k in curr:
                if 'weight' in k or 'bias' in k:  # trainable params
                    diff = curr[k] - prev[k]
                    diff_vec.append(diff.view(-1))

            if not diff_vec:
                dct_reps.append(np.zeros(10))  # dummy
                continue

            full_diff = torch.cat(diff_vec).numpy()

            # Apply DCT (Type 2 is standard)
            # For speed, we only take first X coefficients (Low Frequency)
            # Paper mentions 2D DCT on tensors, but flattening 1D is a common approx for general weights
            # To be precise to paper: "2-D DCT of a signal S... low frequency components"
            # Here we approximate by taking low freq of the flattened vector
            dct_full = scipy.fftpack.dct(full_diff, type=2, norm='ortho')
            # Heuristic: Keep top 10% lowest frequencies
            cutoff = max(1, int(len(dct_full) * 0.1))
            dct_low = dct_full[:cutoff]
            dct_reps.append(dct_low)

        # Compute Pairwise Euclidean Distances
        for t in range(N):
            dists = []
            for j in range(N):
                if t == j:
                    continue
                # Euclidean distance
                d = np.linalg.norm(dct_reps[t] - dct_reps[j])
                dists.append(d)

            # Score = sum of distances to N/2 + 1 closest neighbors
            dists.sort()
            k = int(N / 2) + 1
            scores[t] = sum(dists[:k])

        return scores

    def _compute_rotational_scores(self, N):
        """
        Implements Rotational Distance (Appendix H).
        1. Angular Displacement theta(t) = arctan(B_t)
           (Approximated via flattened vector angle for robustness)
        2. Angular Velocity w(t)
        3. Rotational Distance RD
        """
        # Note: Appendix H describes a complex tensor reshaping method.
        # For this implementation, we use a robust vector-angle interpretation
        # which is mathematically equivalent to "orientation change".

        rd_values = []

        for t in range(N):
            curr_state = self.backbone_history[t]

            # Flatten params to represent B_t
            curr_vec = []
            for k in curr_state:
                curr_vec.append(curr_state[k].view(-1))
            B_t = torch.cat(curr_vec).float()

            # 1. Angular Displacement
            # In high dim, "angle" is relative to origin or an axis.
            # Simple interpretation: The direction of the vector.
            # We compare direction change between t and t-1.

            if t == 0:
                rd_values.append(0.0)
                continue

            prev_vec = []
            for k in self.backbone_history[t-1]:
                prev_vec.append(self.backbone_history[t-1][k].view(-1))
            B_prev = torch.cat(prev_vec).float()

            # Cosine Similarity is a proxy for angular change
            cos_sim = torch.nn.functional.cosine_similarity(
                B_t.unsqueeze(0), B_prev.unsqueeze(0))

            # Angle in radians = arccos(cosine_similarity)
            # Note: The paper uses arctan(B_t) on transformed coords.
            # arccos(sim) gives the angle BETWEEN updates, which is the "displacement".
            theta_diff = torch.acos(torch.clamp(
                cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)).item()

            # 2. Angular Velocity (dt = 1)
            omega = theta_diff / 1.0

            # 3. RD
            rd = omega / (2 * np.pi)
            rd_values.append(rd)

        # Pairwise comparison of RDs
        scores = np.zeros(N)
        for t in range(N):
            diffs = []
            for j in range(N):
                diffs.append(abs(rd_values[t] - rd_values[j]))

            # Score = sum of N/2 + 1 smallest diffs
            diffs.sort()
            k = int(N / 2) + 1
            scores[t] = sum(diffs[:k])

        return scores

    def restore_model(self, backbone, index):
        """Loads the state dict from history at index into backbone"""
        if 0 <= index < len(self.backbone_history):
            backbone.load_state_dict(self.backbone_history[index])
            print(
                f"!!! SafeSplit Triggered: Rolled back to model index {index} !!!")
        return backbone
