import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper
from vqgan_common import VQGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = "./vqgan_decoder_best.pt"
VQVAE_WEIGHTS = "./vqgan_best.pth"

GLASSES = 3
HEAD = 220
BODY = 8

NUM_EMBEDDINGS = 256
EMBEDDING_DIM = 32
NUM_IMAGE_TOKENS = 64
NUM_SAMPLES = 5

TEMPERATURE = 0.1e-7

vqvae = VQGAN()
vqvae.load_state_dict(torch.load(VQVAE_WEIGHTS, map_location=device))
vqvae.to(device)
vqvae.eval()

checkpoint = torch.load(CHECKPOINT, map_location=device)
offsets = checkpoint["offsets"]

offset_glasses = offsets["offset_glasses"]
offset_head = offsets["offset_head"]
offset_body = offsets["offset_body"]
offset_image = offsets["offset_image"]
total_vocab_size = offsets["total_vocab_size"]

decoder = Decoder(dim=256, depth=6, heads=8, attn_flash=True)
transformer_net = TransformerWrapper(num_tokens=total_vocab_size, max_seq_len=67, attn_layers=decoder)
model_ar = AutoregressiveWrapper(transformer_net).to(device)
model_ar.load_state_dict(checkpoint["model_state_dict"])
model_ar.eval()

c1 = GLASSES + offset_glasses
c2 = HEAD + offset_head
c3 = BODY + offset_body

conditions = torch.tensor([[c1, c2, c3]] * NUM_SAMPLES, dtype=torch.long, device=device)

with torch.no_grad():
    generated = model_ar.generate(
        conditions,
        NUM_IMAGE_TOKENS,
        temperature=TEMPERATURE
    )

if generated.shape[1] >= NUM_IMAGE_TOKENS:
    image_tokens = generated[:, -NUM_IMAGE_TOKENS:]

raw_tokens = image_tokens - offset_image

with torch.no_grad():
    z_q = vqvae.vq_vae.embeddings(raw_tokens)
    z_q = z_q.view(NUM_SAMPLES, 8, 8, EMBEDDING_DIM)
    z_q = z_q.permute(0, 3, 1, 2).contiguous()
    images = vqvae.decoder(z_q).cpu()

fig, axes = plt.subplots(1, NUM_SAMPLES, figsize=(3 * NUM_SAMPLES, 3))
if NUM_SAMPLES == 1:
    axes = [axes]

for i in range(NUM_SAMPLES):
    img = images[i].permute(1, 2, 0).numpy()
    rgb = img[:, :, :3]
    axes[i].imshow(rgb)
    axes[i].axis("off")
    axes[i].set_title(f"Próbka {i+1}")

plt.suptitle(f"Glasses={GLASSES}, Head={HEAD}, Body={BODY}")
plt.tight_layout()
plt.show()