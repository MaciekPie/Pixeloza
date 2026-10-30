import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from x_transformers import TransformerWrapper, Encoder

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
def get_all_image_tokens(vqvae_model, images_tensor, batch_size=256):
    all_tokens = []

    for i in range(0, len(images_tensor), batch_size):
        batch = images_tensor[i:i+batch_size].to(device)
        z = vqvae_model.encoder(batch)
        _, _, min_indices = vqvae_model.vq_vae(z) # find the closest tokens
        tokens = min_indices.view(batch.shape[0], -1) # squash other dimensions, other than images themselves
        all_tokens.append(tokens.cpu()) # now it can be put in cpu to release vram memory

    return torch.cat(all_tokens, dim=0) # merge all tokens

raw_img_tokens = get_all_image_tokens(model, images) # shape: [49859, 64]

c1 = features[:, 0] + offset_glasses
c2 = features[:, 1] + offset_head
c3 = features[:, 2] + offset_body

conditions = torch.stack([
    c1,
    c2,
    c3
], dim=1)

img_tokens = raw_img_tokens + offset_image

train_dataset = TensorDataset(
    conditions,
    img_tokens
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=True
)

MASK_PROB = 0.15
DIM = 256

MASK_TOKEN_ID = total_vocab_size
total_vocab_size += 1

condition_embedding = nn.Embedding(
    total_vocab_size,
    DIM
).to(device)

encoder = Encoder(
    dim=DIM,
    depth=6,
    heads=8,
    attn_flash=True,
    cross_attend=True
)

transformer_net = TransformerWrapper(
    num_tokens=total_vocab_size,
    max_seq_len=64,
    attn_layers=encoder
).to(device)

optimizer = optim.AdamW(
    list(transformer_net.parameters()) +
    list(condition_embedding.parameters()),
    lr=3e-4,
    weight_decay=0.01
)

epochs = 200
val_batch_size = 512

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-5
)

use_bfloat16 = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=not use_bfloat16)

total_samples = len(train_dataset)
train_size = int(0.9 * total_samples)
val_size = total_samples - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    shuffle=False
)

def save(save_path="./best_encoder.pt", epoch=None, val_loss=None):
    checkpoint = {
        "transformer_state_dict": transformer_net.state_dict(),
        "condition_embedding_state_dict": condition_embedding.state_dict(),

        "offsets": {
            "offset_glasses": offset_glasses,
            "offset_head": offset_head,
            "offset_body": offset_body,
            "offset_image": offset_image,
            "total_vocab_size": total_vocab_size,
            "mask_token_id": MASK_TOKEN_ID
        },

        "num_features": {
            "num_glasses": num_glasses,
            "num_head": num_head,
            "num_body": num_body
        },

        "model_config": {
            "dim": DIM,
            "depth": 6,
            "heads": 8,
            "max_seq_len": NUM_EMBEDDINGS
        },

        "epoch": epoch,
        "val_loss": val_loss
    }

    torch.save(checkpoint, save_path)
    print(f"\nWagi i metadane zapisane w: {save_path}")

best_val_loss = float("inf")

transformer_net.train()
condition_embedding.train()

for epoch in range(epochs):
    total_loss = 0.0

    pbar = tqdm(
        train_loader,
        desc=f"Epoka {epoch+1}/{epochs}",
        leave=True
    )

    for conditions, img_tokens in pbar:
        conditions = conditions.to(device)
        img_tokens = img_tokens.to(device)

        mask = torch.rand(
            img_tokens.shape,
            device=device
        ) < MASK_PROB

        masked_img_tokens = img_tokens.clone()
        masked_img_tokens[mask] = MASK_TOKEN_ID

        condition_context = condition_embedding(conditions)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = transformer_net(
                masked_img_tokens,
                context=condition_context
            )

            loss = F.cross_entropy(
                logits[mask],
                img_tokens[mask]
            )

        if use_bfloat16:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(transformer_net.parameters()) +
                list(condition_embedding.parameters()),
                1.0
            )
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(transformer_net.parameters()) +
                list(condition_embedding.parameters()),
                1.0
            )
            scaler.step(optimizer)
            scaler.update()

        current_loss = loss.item()
        total_loss += current_loss
        current_lr = optimizer.param_groups[0]['lr']

        pbar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "lr": f"{current_lr:.2e}",
            "masked": f"{mask.float().mean().item():.2%}"
        })

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    print(f"--> Epoka {epoch+1}/{epochs} zakonczona | Sredni Train Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    if (epoch + 1) % 5 == 0:
        transformer_net.eval()
        condition_embedding.eval()

        total_val_loss = 0.0

        with torch.no_grad():
            for conditions, img_tokens in val_loader:
                conditions = conditions.to(device)
                img_tokens = img_tokens.to(device)

                mask = torch.rand(
                    img_tokens.shape,
                    device=device
                ) < MASK_PROB

                masked_img_tokens = img_tokens.clone()
                masked_img_tokens[mask] = MASK_TOKEN_ID

                condition_context = condition_embedding(conditions)

                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = transformer_net(
                        masked_img_tokens,
                        context=condition_context
                    )

                    loss = F.cross_entropy(
                        logits[mask],
                        img_tokens[mask]
                    )

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save("./vae_encoder.pt", epoch=epoch+1, val_loss=best_val_loss)
            print(f"Nowy najlepszy model (val_loss={best_val_loss:.4f})")

        transformer_net.train()
        condition_embedding.train()