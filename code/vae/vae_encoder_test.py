import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from x_transformers import TransformerWrapper, Encoder

CHECKPOINT = "./vae_encoder.pt"
VQVAE_WEIGHTS = "./model_VQ_VAE.pth"

GLASSES = 7
HEAD = 219
BODY = 8

NUM_EMBEDDINGS = 64
EMBEDDING_DIM = 64
DIM = 256
STEPS = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.vq_vae = nn.Embedding(NUM_EMBEDDINGS, EMBEDDING_DIM)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 4, 4, 2, 1),
            nn.Sigmoid()
        )


vqvae = VQVAE().to(device)

vq_state = torch.load(VQVAE_WEIGHTS, map_location=device)

vqvae.vq_vae.weight.data.copy_(
    vq_state["vq_vae.embeddings.weight"]
)

decoder_state = {
    k.replace("decoder.", ""): v
    for k, v in vq_state.items()
    if k.startswith("decoder.")
}

vqvae.decoder.load_state_dict(decoder_state)
vqvae.eval()


checkpoint = torch.load(CHECKPOINT, map_location=device)
offsets = checkpoint["offsets"]

condition_embedding = nn.Embedding(
    offsets["total_vocab_size"],
    DIM
).to(device)

transformer = TransformerWrapper(
    num_tokens=offsets["total_vocab_size"],
    max_seq_len=NUM_EMBEDDINGS,
    attn_layers=Encoder(
        dim=DIM,
        depth=6,
        heads=8,
        attn_flash=True,
        cross_attend=True
    )
).to(device)

transformer.load_state_dict(
    checkpoint["transformer_state_dict"]
)

condition_embedding.load_state_dict(
    checkpoint["condition_embedding_state_dict"]
)

transformer.eval()
condition_embedding.eval()


@torch.no_grad()
def generate(glasses, head, body):
    conditions = torch.tensor([[
        glasses + offsets["offset_glasses"],
        head + offsets["offset_head"],
        body + offsets["offset_body"]
    ]], device=device)

    context = condition_embedding(conditions)

    tokens = torch.full(
        (1, NUM_EMBEDDINGS),
        offsets["mask_token_id"],
        dtype=torch.long,
        device=device
    )

    for step in range(STEPS):
        logits = transformer(
            tokens,
            context=context
        )

        logits[:, :, :offsets["offset_image"]] = -float("inf")
        logits[:, :, offsets["mask_token_id"]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)

        prediction = probs.argmax(dim=-1)

        confidence = probs.gather(
            -1,
            prediction.unsqueeze(-1)
        ).squeeze(-1)

        masked = tokens == offsets["mask_token_id"]
        remaining = masked.sum().item()

        if remaining == 0:
            break

        reveal = max(
            1,
            remaining // (STEPS - step)
        )

        confidence[~masked] = -1

        _, positions = torch.topk(
            confidence,
            k=min(reveal, remaining),
            dim=1
        )

        tokens.scatter_(
            1,
            positions,
            prediction.gather(1, positions)
        )

    return tokens - offsets["offset_image"]


with torch.no_grad():
    tokens = generate(GLASSES, HEAD, BODY)

    z = vqvae.vq_vae(tokens)
    z = z.view(1, 8, 8, EMBEDDING_DIM)
    z = z.permute(0, 3, 1, 2)

    image = vqvae.decoder(z)[0, :3]

plt.imshow(image.permute(1, 2, 0).cpu())
plt.axis("off")
plt.show()