import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

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

GLASSES = 7
HEAD = 220
BODY = 8

NUM_EMBEDDINGS = 256
EMBEDDING_DIM = 32
NUM_IMAGE_TOKENS = 64

CHECKPOINT = "./vqgan_decoder.pt"

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

    z = model.vq_vae.embeddings(tokens)
    z = z.view(1, 8, 8, EMBEDDING_DIM)
    z = z.permute(0, 3, 1, 2)

    image = model.decoder(z)[0, :3]

plt.imshow(image.permute(1, 2, 0).cpu())
plt.axis("off")
plt.show()