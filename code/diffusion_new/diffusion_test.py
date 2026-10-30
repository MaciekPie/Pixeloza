import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusion_common import LinearNoiseScheduler, UNet32

@torch.no_grad()
def generate_images_cfg(model, scheduler, cond, cfg_scale=2.5, device="cuda"):
    model.eval()
    n = cond.shape[0]
    x = torch.randn(n, model.in_channels, 32, 32, device=device)
    drop_mask = torch.ones(n, dtype=torch.bool, device=device)

    for i in reversed(range(scheduler.timesteps)):
        t = torch.full((n,), i, device=device, dtype=torch.long)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            noise_cond = model(x, t, cond, drop_cond_mask=None)
            noise_uncond = model(x, t, cond, drop_cond_mask=drop_mask)
        predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        alpha = scheduler.alpha[i]
        alpha_cumprod = scheduler.alpha_cumprod[i]

        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
        )

        if i > 0:
            sigma = torch.sqrt(scheduler.posterior_variance[i])
            x = x + sigma * torch.randn_like(x)

    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load("../processed_images_features.pt", map_location="cpu")
    images = data["images"]
    features = data["features"].long()

    num_glasses = features[:, 0].max().item() + 1
    num_head    = features[:, 1].max().item() + 1
    num_body    = features[:, 2].max().item() + 1

    scheduler = LinearNoiseScheduler(timesteps=1000).to(device)
    model = UNet32(num_glasses, num_head, num_body, in_channels=4, time_emb_dim=128, base=64).to(device)
    model.load_state_dict(torch.load("./diffusion_model_final.pth", map_location=device))

    original = images[:10].to(device)
    cond = features[:10].to(device)

    generated = generate_images_cfg(model, scheduler, cond, cfg_scale=2.5, device=device)

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))

    for i in range(10):
        orig_img = original[i].permute(1, 2, 0).cpu().numpy()
        gen_img = generated[i].permute(1, 2, 0).cpu().numpy()

        axes[0, i].imshow(orig_img)
        axes[0, i].axis("off")

        axes[1, i].imshow(gen_img)
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original", fontsize=12, loc='left')
    axes[1, 0].set_title("Wygenerowane (CFG 2.5)", fontsize=12, loc='left')

    plt.tight_layout()
    plt.show()