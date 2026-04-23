import torch

ckpt_path = "stories260K.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu")

# --- Extract state_dict (same as your logic) ---
state_dict = None

if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
else:
    try:
        state_dict = checkpoint.state_dict()
    except:
        print("Could not extract state_dict")

# --- Compute min/max ---
if state_dict is not None:
    global_min = float("inf")
    global_max = float("-inf")

    print("\n=== Per-layer min/max ===")

    for name, param in state_dict.items():
        if not torch.is_tensor(param):
            continue  # skip non-tensors

        layer_min = param.min().item()
        layer_max = param.max().item()

        print(f"{name:50} min={layer_min:.6f}  max={layer_max:.6f}")

        # Update global stats
        global_min = min(global_min, layer_min)
        global_max = max(global_max, layer_max)

    print("\n=== Overall ===")
    print(f"Global min: {global_min:.6f}")
    print(f"Global max: {global_max:.6f}")
