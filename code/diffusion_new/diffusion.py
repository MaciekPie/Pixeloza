import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusion_common import PixelArtDataset, LinearNoiseScheduler, UNet32


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Start treningu na: {device}")

    data = torch.load("./processed_images_features.pt", map_location="cpu")
    images_tensor = data["images"]
    features_tensor = data["features"].long()

    num_glasses = features_tensor[:, 0].max().item() + 1
    num_head = features_tensor[:, 1].max().item() + 1
    num_body = features_tensor[:, 2].max().item() + 1

    dataset = PixelArtDataset(images_tensor, features_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )

    scheduler = LinearNoiseScheduler(timesteps=1000).to(device)

    model = UNet32(
        num_glasses,
        num_head,
        num_body,
        in_channels=4,
        time_emb_dim=128,
        base=64
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device == "cuda")
    )

    epochs = 250

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for images, cond in dataloader:
            images = images.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            t = torch.randint(
                0,
                scheduler.timesteps,
                (images.shape[0],),
                device=device
            ).long()

            x_t, noise = scheduler.add_noise(images, t)

            drop_cond_mask = (
                torch.rand(images.shape[0], device=device) < 0.15
            )

            with torch.amp.autocast(
                "cuda",
                enabled=(device == "cuda")
            ):
                predicted_noise = model(
                    x_t,
                    t,
                    cond,
                    drop_cond_mask=drop_cond_mask
                )

                loss = F.mse_loss(predicted_noise, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(
            f"Epoka {epoch:03d}/{epochs} | Loss: {avg_loss:.5f}"
        )

        if epoch % 25 == 0:
            torch.save(
                model.state_dict(),
                f"./diffusion_model_epoch_{epoch}.pth"
            )

    torch.save(
        model.state_dict(),
        "./diffusion_model_final.pth"
    )

    print("Trening zakończony.")