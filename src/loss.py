import torch
import torch.nn as torch_nn
import torch.autograd as autograd

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculates the gradient penalty for WGAN-GP."""
    alpha = torch.rand((real_samples.size(0), 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    fake = torch.ones((real_samples.size(0), 1, d_interpolates.size(2)), device=device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class SpectralLoss(torch_nn.Module):
    def __init__(self):
        super(SpectralLoss, self).__init__()
        
    def forward(self, pred, target):
        # Calculate FFT amplitude difference
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        return torch_nn.functional.l1_loss(pred_amp, target_amp)

def compute_l1_hole_loss(pred, target, mask):
    """L1 loss only on the missing parts (where mask == 0)."""
    hole_mask = 1 - mask
    pred_holes = pred * hole_mask
    target_holes = target * hole_mask
    
    # Avoid zero division
    num_missing = hole_mask.sum() + 1e-8
    loss = torch_nn.functional.l1_loss(pred_holes, target_holes, reduction='sum') / num_missing
    return loss
