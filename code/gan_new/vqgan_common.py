import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# uczy sie lepiej szczegółów, co przy piksel arcie jest sporo pomoglo
class ResBlock(nn.Module):
    num_groups = 8
    neg_slope_a = 0.2
    kernel = 3

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(self.num_groups, channels),             # dzieli na n group i normalizuje po kanale, bardziej stabilne przy mniejszych batchach
            nn.LeakyReLU(self.neg_slope_a, inplace=True),        # wprowadza nieliniowosc, (poza tym prawie wszystko jest tez lepsze niz relu bo nie zabija ujemnych wartosci potrzebnych do nauki jednak)
            nn.Conv2d(channels, channels, self.kernel, 1, 1),    # konwolucja tworzy nowe feature map na podstawie dokladnie blokow kernel X kernel, uczy sie szczegółów
            nn.GroupNorm(self.num_groups, channels),
            nn.LeakyReLU(self.neg_slope_a, inplace=True),
            nn.Conv2d(channels, channels, self.kernel, 1, 1)     # znowu to samo
        )

    def forward(self, x):
        return x + self.block(x) # zwraca poprawke a nie nowy model


# to samo, zmienione (np. na funkcje torcha) dla czytelnosci
# zmniejzony tylko commitment, sprawdzony testowo, wydaje sie ze dla bardziej skomplikowanego modelu i mocniejszej nauki, mozna poluźnić troche enkoder/dekoder
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=256, embedding_dim=32, commitment_cost=0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)

        distances = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(z, self.embeddings.weight.t())
        )

        indices = distances.argmin(1)
        z_q = self.embeddings(indices).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        vq_loss = F.mse_loss(z_q, z_e.detach()) + self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        z_q = z_e + (z_q - z_e).detach()
        return z_q, vq_loss, indices.view(B, H, W)


class VQGAN(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 64
        in_channels = 4
        out_channels = 32

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, channels, 4, 2, 1),
            ResBlock(channels),
            nn.Conv2d(channels, channels, 4, 2, 1),
            ResBlock(channels),
            nn.Conv2d(channels, out_channels, 3, 1, 1)
        )

        self.vq_vae = VectorQuantizer(num_embeddings=256, embedding_dim=32)

        self.decoder = nn.Sequential(
            nn.Conv2d(out_channels, channels, 3, 1, 1),
            ResBlock(channels),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels, channels, 3, 1, 1),
            ResBlock(channels),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(ResBlock.num_groups, channels),
            nn.LeakyReLU(ResBlock.neg_slope_a, inplace=True),
            nn.Conv2d(channels, in_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq_vae(z)
        return self.decoder(z_q), vq_loss, indices


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Dodany Spectral Norm zapobiega eksplozji gradientu w dyskryminatorze
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(4, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 1, 3, 1, 1))
        )

    def forward(self, x):
        return self.net(x)