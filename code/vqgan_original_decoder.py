import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=32, commitment_cost=0.5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)
        distances = (
            z_e_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_e_flat @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        z_q = self.embeddings(indices).view(B, H, W, C).permute(0, 3, 1, 2)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        z_q = z_e + (z_q - z_e).detach()
        return z_q, vq_loss, indices.view(B, H, W)


class VQGAN(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=64, latent_dim=32, num_embeddings=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_dim, 3, stride=1, padding=1),
        )
        self.vq_vae = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq_vae(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices


NUM_EMBEDDINGS = 256

model = VQGAN(in_channels=4, hidden_dim=64, latent_dim=32, num_embeddings=NUM_EMBEDDINGS).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
model.load_state_dict(torch.load("./vqgan.pth", map_location=device))
model.eval()
print("Model wczytany.")    


# ------------------------------------------

tensor_data = torch.load("../processed_images_features.pt", map_location='cpu')

images = tensor_data["images"]      # Shape: [49859, 4, 32, 32]
features = tensor_data["features"]  # Shape: [49859, 3] (glasses, head, body)

num_glasses = features[:, 0].max().item() + 1
num_head    = features[:, 1].max().item() + 1
num_body    = features[:, 2].max().item() + 1

#unique tokens in codebook
offset_glasses = 0
offset_head    = offset_glasses + num_glasses
offset_body    = offset_head + num_head
offset_image   = offset_body + num_body

total_vocab_size = offset_image + NUM_EMBEDDINGS

@torch.no_grad()
def get_all_image_tokens(vq_vqgan, images_tensor, batch_size=256):
    all_tokens = []

    for i in range(0, len(images_tensor), batch_size):
        batch = images_tensor[i:i+batch_size].to(device)
        z = vq_vqgan.encoder(batch)
        _, _, min_indices = vq_vqgan.vq_vae(z) # find the closest tokens
        tokens = min_indices.view(batch.shape[0], -1) # squash other dimensions, other than images themselves
        all_tokens.append(tokens.cpu()) # now it can be put in cpu to release vram memory

    return torch.cat(all_tokens, dim=0) # merge all tokens

raw_img_tokens = get_all_image_tokens(model, images) # shape: [49859, 64]

c1 = features[:, 0] + offset_glasses
c2 = features[:, 1] + offset_head
c3 = features[:, 2] + offset_body
img_tokens = raw_img_tokens + offset_image # apply different offset to image tokens, so they are different and dont collide with features. Remember we only teach here the transformer not vq_vqgan

full_sequences = torch.cat([
    c1.unsqueeze(1), # now [49859, 1] we create feature "glass" dimension, in glass dim, there are 49859 records
    c2.unsqueeze(1), # now [49859, 1]
    c3.unsqueeze(1), # now [49859, 1]
    img_tokens # [49859, 64]
], dim=1).long()

decoder = Decoder(
    dim = 256,
    depth = 6,
    heads = 8,
    attn_flash = True
)

transformer_net = TransformerWrapper(
    num_tokens = total_vocab_size,
    max_seq_len = 67,
    attn_layers = decoder
)

model_ar = AutoregressiveWrapper(transformer_net).to(device)

full_sequences = full_sequences.to(device)

total_samples = len(full_sequences)
train_size = int(0.9 * total_samples)

indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42)).to(device)
train_data = full_sequences[indices[:train_size]]
val_data = full_sequences[indices[train_size:]]

use_bfloat16 = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

epochs = 200
batch_size = 1024
val_batch_size = 512

optimizer = optim.AdamW(model_ar.parameters(), lr=3e-4, weight_decay=0.01)
# Harmonogram tempa nauki (płynne obniżanie LR do 1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
scaler = torch.amp.GradScaler("cuda", enabled=not use_bfloat16)

def save(save_path="./best_decoder.pt", epoch=None, val_loss=None):
    checkpoint = {
        "model_state_dict": model_ar.state_dict(),
        "offsets": {
            "offset_glasses": offset_glasses,
            "offset_head": offset_head,
            "offset_body": offset_body,
            "offset_image": offset_image,
            "total_vocab_size": total_vocab_size
        },
        "num_features": {
            "num_glasses": num_glasses,
            "num_head": num_head,
            "num_body": num_body
        },
        "epoch": epoch,
        "val_loss": val_loss
    }
    torch.save(checkpoint, save_path)
    print(f"\nWagi i metadane zapisane w: {save_path}")

best_val_loss = float("inf")

for epoch in range(epochs):
    model_ar.train()
    total_loss = 0.0
    
    permutation = torch.randperm(train_data.size(0), device=device)
    batch_indices = range(0, train_data.size(0), batch_size)
    
    pbar = tqdm(batch_indices, desc=f"Epoka {epoch+1}/{epochs}", leave=True)

    for i in pbar:
        batch_idx = permutation[i:i + batch_size]
        seq = train_data[batch_idx]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            loss = model_ar(seq)

        if use_bfloat16:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_ar.parameters(), 1.0)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_ar.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        current_loss = loss.item()
        total_loss += current_loss
        current_lr = optimizer.param_groups[0]['lr']

        pbar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "lr": f"{current_lr:.2e}"
        })

    scheduler.step()

    avg_loss = total_loss / len(batch_indices)
    print(f"--> Epoka {epoch+1}/{epochs} zakonczona | Sredni Train Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    if (epoch + 1) % 5 == 0:
        model_ar.eval()
        total_val_loss = 0.0
        val_batch_indices = range(0, val_data.size(0), val_batch_size)

        with torch.no_grad():
            for i in val_batch_indices:
                seq = val_data[i:i + val_batch_size]

                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = model_ar(seq)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_batch_indices)
        print(f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save("./vq_vqgan_decoder.pt", epoch=epoch+1, val_loss=best_val_loss)
            print(f"Nowy najlepszy model (val_loss={best_val_loss:.4f})")