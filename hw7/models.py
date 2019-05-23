import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, enc_channel=16, dec_channel=64, latent_dim=32):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # 3*32*32 -> 16*16*16
            nn.Conv2d(in_channels=3, out_channels=enc_channel, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(True),
            # 16*16*16 -> 32*8*8
            nn.Conv2d(in_channels=enc_channel, out_channels=enc_channel*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(enc_channel*2),
            nn.LeakyReLU(True),
            # 32*8*8 -> 64*4*4
            nn.Conv2d(in_channels=enc_channel*2, out_channels=enc_channel*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(enc_channel*4),
            nn.LeakyReLU(True),
            # 64*4*4 -> 128*2*2
            nn.Conv2d(in_channels=enc_channel*4, out_channels=enc_channel*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(enc_channel*8),
            nn.LeakyReLU(True)
        )

        self.enc_fc = nn.Sequential(
            nn.Linear(in_features=enc_channel*8*2*2, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(True),
            nn.Linear(in_features=128, out_features=latent_dim)
        )

        self.dec_fc = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=128),
            nn.ReLU(True),
            nn.Linear(in_features=128, out_features=enc_channel*8*2*2),
            nn.BatchNorm1d(enc_channel*8*2*2),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            # 128*2*2 -> 64*4*4
            nn.ConvTranspose2d(in_channels=enc_channel*8, out_channels=dec_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dec_channel),
            nn.ReLU(True),
            # 64*4*4 -> 32*8*8
            nn.ConvTranspose2d(in_channels=dec_channel, out_channels=dec_channel//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dec_channel//2),
            nn.ReLU(True),
            # 32*8*8 -> 16*16*16
            nn.ConvTranspose2d(in_channels=dec_channel//2, out_channels=dec_channel//4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dec_channel//4),
            nn.ReLU(True),
            # 16*16*16 -> 3*32*32
            nn.ConvTranspose2d(in_channels=dec_channel//4, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.enc_fc(x)
        x = self.dec_fc(x)
        x = x.contiguous().view(x.size(0), -1, 2, 2)
        x = self.decoder(x)
        return x