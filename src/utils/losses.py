"""
Loss functions for super-resolution training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 variant)
    More robust than L1 and L2 for super-resolution
    """
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return torch.mean(loss)


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG19 features
    Measures high-level feature similarity
    """
    def __init__(self, layer_weights=None):
        super(PerceptualLoss, self).__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # Extract specific layers for perceptual loss
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18]) # relu3_4
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27])# relu4_4
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Layer weights (default: equal weight to all layers)
        if layer_weights is None:
            self.layer_weights = [1.0, 1.0, 1.0, 1.0]
        else:
            self.layer_weights = layer_weights
        
        self.criterion = nn.L1Loss()
    
    def normalize_batch(self, batch):
        """Normalize batch for VGG (ImageNet stats)"""
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (batch - mean) / std
    
    def forward(self, pred, target):
        # Normalize inputs
        pred = self.normalize_batch(pred)
        target = self.normalize_batch(target)
        
        # Extract features
        pred_relu1 = self.slice1(pred)
        pred_relu2 = self.slice2(pred_relu1)
        pred_relu3 = self.slice3(pred_relu2)
        pred_relu4 = self.slice4(pred_relu3)
        
        target_relu1 = self.slice1(target)
        target_relu2 = self.slice2(target_relu1)
        target_relu3 = self.slice3(target_relu2)
        target_relu4 = self.slice4(target_relu3)
        
        # Calculate perceptual loss
        loss = 0
        loss += self.layer_weights[0] * self.criterion(pred_relu1, target_relu1)
        loss += self.layer_weights[1] * self.criterion(pred_relu2, target_relu2)
        loss += self.layer_weights[2] * self.criterion(pred_relu3, target_relu3)
        loss += self.layer_weights[3] * self.criterion(pred_relu4, target_relu4)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for super-resolution:
    L1 + Perceptual + optional adversarial
    """
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, use_perceptual=True):
        super(CombinedLoss, self).__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual
        
        # Charbonnier is smoother than L1
        self.pixel_loss = CharbonnierLoss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred, target):
        # Pixel loss
        loss_pixel = self.pixel_loss(pred, target)
        total_loss = self.l1_weight * loss_pixel
        
        loss_dict = {
            'pixel': loss_pixel.item()
        }
        
        # Perceptual loss
        if self.use_perceptual:
            loss_perceptual = self.perceptual_loss(pred, target)
            total_loss += self.perceptual_weight * loss_perceptual
            loss_dict['perceptual'] = loss_perceptual.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test losses
    print("🧪 Testing loss functions...")
    
    pred = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, 3, 256, 256)
    
    # Test Charbonnier Loss
    char_loss = CharbonnierLoss()
    loss = char_loss(pred, target)
    print(f"✅ Charbonnier Loss: {loss.item():.6f}")
    
    # Test Perceptual Loss
    print("Loading VGG19 for perceptual loss...")
    perc_loss = PerceptualLoss()
    loss = perc_loss(pred, target)
    print(f"✅ Perceptual Loss: {loss.item():.6f}")
    
    # Test Combined Loss
    combined = CombinedLoss()
    total_loss, loss_dict = combined(pred, target)
    print(f"✅ Combined Loss: {total_loss.item():.6f}")
    print(f"   - Pixel: {loss_dict['pixel']:.6f}")
    print(f"   - Perceptual: {loss_dict['perceptual']:.6f}")
