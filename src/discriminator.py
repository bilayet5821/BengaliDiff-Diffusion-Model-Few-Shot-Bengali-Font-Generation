# src/discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 1, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 8, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

def discriminator_loss(real_pred, fake_pred, mode='hinge'):
    if mode == 'hinge':
        loss_real = torch.mean(F.relu(1.0 - real_pred))
        loss_fake = torch.mean(F.relu(1.0 + fake_pred))
    elif mode == 'lsgan':
        loss_real = F.mse_loss(real_pred, torch.ones_like(real_pred))
        loss_fake = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
    else:
        raise NotImplementedError(f"Unsupported GAN loss: {mode}")
    return loss_real + loss_fake

def generator_gan_loss(fake_pred, mode='hinge'):
    if mode == 'hinge':
        return -torch.mean(fake_pred)
    elif mode == 'lsgan':
        return F.mse_loss(fake_pred, torch.ones_like(fake_pred))
    else:
        raise NotImplementedError(f"Unsupported GAN loss: {mode}")
