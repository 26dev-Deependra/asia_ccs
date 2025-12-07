import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy

from models.model_vit import ViTHead, ViTBackbone, ViTTail
from utils.data_utils import get_dataloader, Attacker, calculate_accuracy
from safesplit import SafeSplitDefense

# Load Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config['system']['device'])

# --- Initialize Models ---
# Parameters for ViT
img_size = config['model']['image_size']
patch_size = config['model']['patch_size']
dim = config['model']['dim']
depth = config['model']['depth']
heads = config['model']['heads']
mlp_dim = config['model']['mlp_dim']
num_classes = config['model']['num_classes']

# Split Depths
d_head = config['model']['cut_layer_1']
d_backbone = config['model']['cut_layer_2'] - config['model']['cut_layer_1']
d_tail = depth - config['model']['cut_layer_2']

# 1. Server Model (Backbone)
backbone = ViTBackbone(dim, d_backbone, heads, mlp_dim).to(device)
opt_backbone = optim.Adam(backbone.parameters(), lr=config['training']['lr'])

# 2. Client Models (Simulating N clients by re-initializing or loading weights)
# In simulation, we often keep one instance and reset or keep state.
# Here we keep separate state dicts for each client to be realistic.
clients_state = {}
for i in range(config['system']['num_clients']):
    clients_state[i] = {
        'head': ViTHead(img_size, patch_size, dim, d_head, heads, mlp_dim).to(device).state_dict(),
        'tail': ViTTail(dim, d_tail, heads, mlp_dim, num_classes).to(device).state_dict()
    }

# --- Defense & Attacker ---
defense = SafeSplitDefense(config) if config['defense']['active'] else None
attacker = Attacker(config)
criterion = nn.CrossEntropyLoss()

# --- Training Loop ---
train_loader = get_dataloader(config, train=True)
print(f"Starting Training: {config['experiment']['name']}")

for r in range(config['system']['rounds']):
    print(f"\n--- Round {r+1}/{config['system']['rounds']} ---")

    # Iterate over clients sequentially (Standard Split Learning)
    for client_id in range(config['system']['num_clients']):

        # Check if client is malicious
        is_malicious = client_id in config['system']['malicious_clients']
        attack_active = is_malicious and attacker.is_active(r)

        # Load Client Models
        head = ViTHead(img_size, patch_size, dim,
                       d_head, heads, mlp_dim).to(device)
        tail = ViTTail(dim, d_tail, heads, mlp_dim, num_classes).to(device)

        head.load_state_dict(clients_state[client_id]['head'])
        tail.load_state_dict(clients_state[client_id]['tail'])

        # Optimizers (Clients usually train locally for 1 epoch or a few batches)
        opt_head = optim.Adam(head.parameters(), lr=config['training']['lr'])
        opt_tail = optim.Adam(tail.parameters(), lr=config['training']['lr'])

        # Get a batch of data
        try:
            images, labels = next(data_iter)
        except:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        images, labels = images.to(device), labels.to(device)

        # --- ATTACK INJECTION ---
        if attack_active:
            print(f"Client {client_id} performing attack...")
            images, labels = attacker.poison_batch(images, labels)

        # --- U-SHAPED FORWARD PASS ---

        # 1. Client Head Forward
        opt_head.zero_grad()
        client_smashed = head(images)

        # Send smashed data to Server (detach from graph to simulate network boundary, usually requires requires_grad=True)
        server_input = client_smashed.clone().detach().requires_grad_(True)

        # 2. Server Backbone Forward
        opt_backbone.zero_grad()
        server_output = backbone(server_input)

        # Send output to Client
        client_tail_input = server_output.clone().detach().requires_grad_(True)

        # 3. Client Tail Forward
        opt_tail.zero_grad()
        outputs = tail(client_tail_input)
        loss = criterion(outputs, labels)

        # --- BACKWARD PASS ---

        # 1. Client Tail Backward
        loss.backward()
        opt_tail.step()

        # Gradients for server
        grad_for_server = client_tail_input.grad.clone()

        # 2. Server Backbone Backward
        server_output.backward(grad_for_server)
        opt_backbone.step()

        # Gradients for client head
        grad_for_head = server_input.grad.clone()

        # 3. Client Head Backward
        client_smashed.backward(grad_for_head)
        opt_head.step()

        # Save Client State
        clients_state[client_id]['head'] = head.state_dict()
        clients_state[client_id]['tail'] = tail.state_dict()

        # --- DEFENSE CHECK (After training) ---
        if defense:
            # 1. Add current backbone state to history
            defense.update_history(backbone)

            # 2. Run Analysis
            safe_index = defense.get_latest_valid_model_index()

            if safe_index != -1:
                # Malicious behavior detected!
                print(
                    f"Defense Alert: Client {client_id} flagged! Rolling back...")
                backbone = defense.restore_model(backbone, safe_index)
                # In real U-Shaped, we might also need to reject the client's Head/Tail update,
                # but server can't control that. Server only protects the backbone.

        # Acc Log
        acc = calculate_accuracy(outputs, labels)
        if client_id % 5 == 0:
            print(f"Client {client_id} Loss: {loss.item():.4f} Acc: {acc:.4f}")

print("Training Complete.")
