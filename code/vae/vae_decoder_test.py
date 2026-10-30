import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper

CHECKPOINT = "./vae_decoder.pt"
VQVAE_WEIGHTS = "./model_VQ_VAE.pth"

GLASSES = 7
HEAD = 220
BODY = 8

NUM_EMBEDDINGS = 64
EMBEDDING_DIM = 64
NUM_IMAGE_TOKENS = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1)
        )

        self.vq_vae = VectorQuantizer(
            NUM_EMBEDDINGS,
            EMBEDDING_DIM
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 4, 4, 2, 1),
            nn.Sigmoid()
        )


vqvae = VQVAE().to(device)
vqvae.load_state_dict(
    torch.load(VQVAE_WEIGHTS, map_location=device)
)
vqvae.eval()

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

offsets = checkpoint["offsets"]

decoder = Decoder(
    dim=256,
    depth=6,
    heads=8,
    attn_flash=True
)

transformer = AutoregressiveWrapper(
    TransformerWrapper(
        num_tokens=offsets["total_vocab_size"],
        max_seq_len=67,
        attn_layers=decoder
    )
).to(device)

transformer.load_state_dict(
    checkpoint["model_state_dict"]
)

transformer.eval()

conditions = torch.tensor([[
    GLASSES + offsets["offset_glasses"],
    HEAD + offsets["offset_head"],
    BODY + offsets["offset_body"]
]], device=device)

with torch.no_grad():

    tokens = transformer.generate(
        conditions,
        NUM_IMAGE_TOKENS,
        temperature=1e-8
    )

    tokens = tokens - offsets["offset_image"]

    z = vqvae.vq_vae.embeddings(tokens)
    z = z.view(1, 8, 8, EMBEDDING_DIM)
    z = z.permute(0, 3, 1, 2)

    image = vqvae.decoder(z)[0, :3]

plt.imshow(image.permute(1, 2, 0).cpu())
plt.axis("off")
plt.show()