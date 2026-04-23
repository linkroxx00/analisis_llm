import torch

# Load checkpoint
ckpt_path = "stories260K.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu")

print("=== Checkpoint type ===")
print(type(checkpoint))

# If it is a dict, print keys
if isinstance(checkpoint, dict):
    print("\n=== Checkpoint keys ===")
    for k in checkpoint.keys():
        print(k)

# Try to find state_dict automatically
state_dict = None

if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        # sometimes the dict itself is the state_dict
        state_dict = checkpoint
else:
    # If the checkpoint itself is a model
    try:
        state_dict = checkpoint.state_dict()
    except:
        print("Could not extract state_dict")

# Print architecture
if state_dict is not None:
    print("\n=== Model Weights Architecture ===")
    total_params = 0

    for name, param in state_dict.items():
        print(f"{name:50} {tuple(param.shape)}")
        total_params += param.numel()

    print("\nTotal parameters:", total_params)
