import torch
import torch.nn as torch_nn

class Generator(torch.nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=16):
        super(Generator, self).__init__()
        
        # Tiny U-Net
        self.enc1 = torch_nn.Sequential(
            torch_nn.Conv1d(in_channels, features, kernel_size=3, padding=1),
            torch_nn.ReLU(inplace=True),
            torch_nn.Conv1d(features, features*2, kernel_size=3, padding=1, stride=2),
            torch_nn.ReLU(inplace=True)
        )
        
        self.enc2 = torch_nn.Sequential(
            torch_nn.Conv1d(features*2, features*4, kernel_size=3, padding=1, stride=2),
            torch_nn.ReLU(inplace=True)
        )
        
        self.dec1 = torch_nn.Sequential(
            torch_nn.ConvTranspose1d(features*4, features*2, kernel_size=4, padding=1, stride=2),
            torch_nn.ReLU(inplace=True)
        )
        
        self.dec2 = torch_nn.Sequential(
            torch_nn.ConvTranspose1d(features*4, features, kernel_size=4, padding=1, stride=2),
            torch_nn.ReLU(inplace=True)
        )
        
        self.final = torch_nn.Conv1d(features + in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, masked_signal, mask):
        # Input formatting: (B, 2, L)
        x = torch.cat([masked_signal, mask], dim=1)
        
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        # Decode
        d1 = self.dec1(e2)
        # Skip connection 1
        d1_cat = torch.cat([d1, e1], dim=1) 
        
        d2 = self.dec2(d1_cat)
        # Skip connection 2 (Original Input)
        d2_cat = torch.cat([d2, x], dim=1)
        
        out = self.final(d2_cat)
        return out

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=1, features=8):
        super(Discriminator, self).__init__()
        
        # Tiny PatchGAN 1D
        self.model = torch_nn.Sequential(
            torch_nn.Conv1d(in_channels, features, kernel_size=4, stride=2, padding=1),
            torch_nn.LeakyReLU(0.2, inplace=True),
            
            torch_nn.Conv1d(features, features*2, kernel_size=4, stride=2, padding=1),
            torch_nn.LeakyReLU(0.2, inplace=True),
            
            # Output is a sequence of scalars judging different patches
            torch_nn.Conv1d(features*2, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, signal):
        return self.model(signal)
