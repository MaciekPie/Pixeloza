import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from x_transformers import TransformerWrapper, Encoder
from vqgan_common import VQGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EMBEDDINGS = 256
model = VQGAN()
weights_path = "./vqgan_best.pth"

model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)
model.eval()

tensor_data = torch.load("./processed_images_features.pt", map_location='cpu')

images = tensor_data["images"]
features = tensor_data["features"]

num_glasses = features[:, 0].max().item() + 1
num_head    = features[:, 1].max().item() + 1
num_body    = features[:, 2].max().item() + 1

offset_glasses = 0
offset_head    = offset_glasses + num_glasses
offset_body    = offset_head + num_head
offset_image   = offset_body + num_body

total_vocab_size = offset_image + NUM_EMBEDDINGS

@torch.no_grad()
def get_all_image_tokens(vqgan_model, images_tensor, batch_size=256):
    all_tokens = []
    for i in range(0, len(images_tensor), batch_size):
        batch = images_tensor[i:i+batch_size].to(device)
        z = vqgan_model.encoder(batch)
        _, _, min_indices = vqgan_model.vq_vae(z)
        tokens = min_indices.view(batch.shape[0], -1)
        all_tokens.append(tokens.cpu())
    return torch.cat(all_tokens, dim=0)

raw_img_tokens = get_all_image_tokens(model, images)

c1 = features[:, 0] + offset_glasses
c2 = features[:, 1] + offset_head
c3 = features[:, 2] + offset_body

conditions = torch.stack([c1, c2, c3], dim=1)
img_tokens = raw_img_tokens + offset_image

train_dataset = TensorDataset(conditions, img_tokens)

MASK_PROB = 0.15
DIM = 256

MASK_TOKEN_ID = total_vocab_size
total_vocab_size += 1

condition_embedding = nn.Embedding(total_vocab_size, DIM).to(device)

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
    list(transformer_net.parameters()) + list(condition_embedding.parameters()),
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

def save(save_path="./vqgan_encoder.pt", epoch=None, val_loss=None):
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
            "max_seq_len": 64
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

        mask = torch.rand(img_tokens.shape, device=device) < MASK_PROB

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
                list(transformer_net.parameters()) + list(condition_embedding.parameters()),
                1.0
            )
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(transformer_net.parameters()) + list(condition_embedding.parameters()),
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

                mask = torch.rand(img_tokens.shape, device=device) < MASK_PROB

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
            save("./vqgan_encoder.pt", epoch=epoch+1, val_loss=best_val_loss)
            print(f"Nowy najlepszy model (val_loss={best_val_loss:.4f})")

        transformer_net.train()
        condition_embedding.train()