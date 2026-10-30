import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from vqgan_common import VQGAN, PatchDiscriminator

torch.backends.cudnn.benchmark = True

data = torch.load("./processed_images_features.pt", map_location="cpu")
images_tensor = data["images"]

class PixelArtDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

dataset = PixelArtDataset(images_tensor)
train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VQGAN().to(device)
discriminator = PatchDiscriminator().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.9))
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

disc_start_epoch = 10
adv_weight_max = 0.01
num_epochs = 200
best_val = float("inf")

def calc_recon_loss(recon, target):
    # zmieniona waga na alpha, bo sie psul lol
    rgb_loss = F.l1_loss(recon[:, :3], target[:, :3])
    alpha_loss = F.l1_loss(recon[:, 3:], target[:, 3:])
    return rgb_loss + 2.0 * alpha_loss

@torch.no_grad()
def validate():
    model.eval()
    loss = 0
    for batch in val_loader:
        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            recon, _, _ = model(batch)
            loss += calc_recon_loss(recon, batch).item()
    return loss / len(val_loader)

for epoch in range(num_epochs):
    model.train()
    discriminator.train()

    recon_sum = vq_sum = disc_sum = adv_sum = 0
    used_indices = set()
    use_disc = epoch >= disc_start_epoch

    for batch in train_loader:
        batch = batch.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            recon, vq_loss, indices = model(batch)
            recon_loss = calc_recon_loss(recon, batch)

        used_indices.update(torch.unique(indices).tolist())

        if use_disc:
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                real = discriminator(batch)
                fake = discriminator(recon.detach())
                disc_loss = 0.5 * (
                    F.mse_loss(real, torch.ones_like(real)) +
                    F.mse_loss(fake, torch.zeros_like(fake))
                )

            disc_optimizer.zero_grad()
            scaler.scale(disc_loss).backward()
            scaler.step(disc_optimizer)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                adv_loss = F.mse_loss(discriminator(recon), torch.ones_like(real))
                loss = recon_loss + vq_loss + adv_weight_max * adv_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            disc_sum += disc_loss.item()
            adv_sum += adv_loss.item()
        else:
            loss = recon_loss + vq_loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

        scaler.update()

        recon_sum += recon_loss.item()
        vq_sum += vq_loss.item()

    n = len(train_loader)
    print(
        f"Epoka {epoch+1:03d}/{num_epochs} | "
        f"recon: {recon_sum/n:.5f} | "
        f"vq: {vq_sum/n:.5f} | "
        f"disc: {disc_sum/n:.5f} | "
        f"codebook: {len(used_indices)}/256"
    )

    if (epoch + 1) % 5 == 0:
        val_loss = validate()
        print(f"--> VAL Loss: {val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "./vqgan_best.pth")
            print("--> Zapisano model.")