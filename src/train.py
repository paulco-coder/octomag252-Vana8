import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def train_gan(dataset, generator, discriminator, epochs=2, batch_size=16, d_updates=2, g_updates=1, lambda_gp=10, lambda_l1=100, lambda_spectral=1.0, device='cpu'):
    # Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.9))
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Needs the loss modules
    from .loss import compute_gradient_penalty, SpectralLoss, compute_l1_hole_loss
    spectral_loss = SpectralLoss().to(device)
    
    generator.to(device)
    discriminator.to(device)
    
    print(f"Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        for i, (real_imgs, masks, masked_imgs) in enumerate(loader):
            real_imgs = real_imgs.to(device)
            masks = masks.to(device)
            masked_imgs = masked_imgs.to(device)
            
            # --- Train Discriminator ---
            for _ in range(d_updates):
                opt_d.zero_grad()
                
                # Generate fake data
                fake_imgs = generator(masked_imgs, masks)
                
                # Predictions
                real_validity = discriminator(real_imgs)
                fake_validity = discriminator(fake_imgs.detach())
                
                # Wasserstein Loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                
                # Gradient Penalty
                gp = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, device)
                
                # Total D Loss
                d_loss_total = d_loss + lambda_gp * gp
                d_loss_total.backward()
                opt_d.step()
                
            # --- Train Generator ---
            for _ in range(g_updates):
                opt_g.zero_grad()
                
                fake_imgs = generator(masked_imgs, masks)
                fake_validity = discriminator(fake_imgs)
                
                # Adversarial Loss
                g_adv = -torch.mean(fake_validity)
                
                # Continuity L1 Loss (only on holes)
                g_l1 = compute_l1_hole_loss(fake_imgs, real_imgs, masks)
                
                # Spectral Loss
                g_spectral = spectral_loss(fake_imgs, real_imgs)
                
                # Total G Loss
                g_loss_total = g_adv + lambda_l1 * g_l1 + lambda_spectral * g_spectral
                g_loss_total.backward()
                opt_g.step()
                
            if i % max(1, len(loader) // 2) == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(loader)}] "
                      f"[D loss: {d_loss_total.item():.4f}] [G loss: {g_loss_total.item():.4f}] (L1: {g_l1.item():.4f}, Spec: {g_spectral.item():.4f})")
                      
    print("Training finished.")
    return generator
