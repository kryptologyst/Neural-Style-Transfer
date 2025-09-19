"""
Modern Neural Style Transfer implementation with latest techniques.
Includes AdaIN, MSG-Net, and improved optimization strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional
import cv2


class AdaINStyleTransfer(nn.Module):
    """Adaptive Instance Normalization Style Transfer"""
    
    def __init__(self):
        super(AdaINStyleTransfer, self).__init__()
        # Use pre-trained VGG19 as encoder
        vgg = models.vgg19(pretrained=True).features
        self.encoder = nn.Sequential(*list(vgg.children())[:21])  # up to relu4_1
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def adaptive_instance_norm(self, content_feat, style_feat):
        """AdaIN operation"""
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)
        
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)
    
    def calc_mean_std(self, feat, eps=1e-5):
        """Calculate mean and std for AdaIN"""
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
    
    def forward(self, content, style, alpha=1.0):
        content_feat = self.encode(content)
        style_feat = self.encode(style)
        
        # Apply AdaIN
        target_feat = self.adaptive_instance_norm(content_feat, style_feat)
        target_feat = alpha * target_feat + (1 - alpha) * content_feat
        
        return self.decode(target_feat)


class FastNeuralStyleTransfer(nn.Module):
    """Fast Neural Style Transfer with improved architecture"""
    
    def __init__(self, style_weight=1e10, content_weight=1e5):
        super(FastNeuralStyleTransfer, self).__init__()
        self.style_weight = style_weight
        self.content_weight = content_weight
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.vgg = vgg.eval()
        
        # Define layers for style and content
        self.style_layers = ['0', '5', '10', '19', '28']  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        self.content_layers = ['21']  # conv4_2
        
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def get_features(self, image, layers):
        """Extract features from specified layers"""
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[name] = x
        return features
    
    def gram_matrix(self, tensor):
        """Calculate Gram matrix for style loss"""
        b, c, h, w = tensor.size()
        tensor = tensor.view(b * c, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram.div(b * c * h * w)
    
    def style_loss(self, gen_features, style_features):
        """Calculate style loss using Gram matrices"""
        loss = 0
        for layer in self.style_layers:
            gen_gram = self.gram_matrix(gen_features[layer])
            style_gram = self.gram_matrix(style_features[layer])
            loss += F.mse_loss(gen_gram, style_gram)
        return loss
    
    def content_loss(self, gen_features, content_features):
        """Calculate content loss"""
        loss = 0
        for layer in self.content_layers:
            loss += F.mse_loss(gen_features[layer], content_features[layer])
        return loss


class StyleTransferPipeline:
    """Complete style transfer pipeline with modern techniques"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.adain_model = AdaINStyleTransfer().to(device)
        self.fast_model = FastNeuralStyleTransfer().to(device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.denormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def save_image(self, tensor: torch.Tensor, path: str):
        """Save tensor as image"""
        tensor = tensor.cpu().squeeze(0)
        tensor = self.denormalize(tensor)
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL Image
        tensor = tensor.permute(1, 2, 0).numpy()
        image = Image.fromarray((tensor * 255).astype(np.uint8))
        image.save(path)
    
    def transfer_style_adain(self, content_path: str, style_path: str, 
                           output_path: str, alpha: float = 1.0):
        """Perform style transfer using AdaIN"""
        content = self.load_image(content_path)
        style = self.load_image(style_path)
        
        with torch.no_grad():
            output = self.adain_model(content, style, alpha)
        
        self.save_image(output, output_path)
        return output_path
    
    def transfer_style_optimization(self, content_path: str, style_path: str,
                                  output_path: str, steps: int = 500):
        """Perform style transfer using optimization-based method"""
        content = self.load_image(content_path)
        style = self.load_image(style_path)
        
        # Initialize generated image
        generated = content.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([generated], lr=1)
        
        # Get target features
        content_features = self.fast_model.get_features(content, 
                                                       self.fast_model.content_layers)
        style_features = self.fast_model.get_features(style, 
                                                     self.fast_model.style_layers)
        
        def closure():
            optimizer.zero_grad()
            
            gen_features_style = self.fast_model.get_features(generated, 
                                                            self.fast_model.style_layers)
            gen_features_content = self.fast_model.get_features(generated, 
                                                              self.fast_model.content_layers)
            
            style_loss = self.fast_model.style_loss(gen_features_style, style_features)
            content_loss = self.fast_model.content_loss(gen_features_content, content_features)
            
            total_loss = (self.fast_model.style_weight * style_loss + 
                         self.fast_model.content_weight * content_loss)
            
            total_loss.backward()
            return total_loss
        
        # Optimization loop
        for i in range(steps):
            optimizer.step(closure)
            if i % 50 == 0:
                print(f"Step {i}/{steps}")
        
        self.save_image(generated, output_path)
        return output_path
    
    def batch_style_transfer(self, content_images: List[str], 
                           style_images: List[str], 
                           output_dir: str, method: str = 'adain'):
        """Batch process multiple images"""
        results = []
        
        for i, (content_path, style_path) in enumerate(zip(content_images, style_images)):
            output_path = f"{output_dir}/stylized_{i}.jpg"
            
            if method == 'adain':
                result = self.transfer_style_adain(content_path, style_path, output_path)
            else:
                result = self.transfer_style_optimization(content_path, style_path, output_path)
            
            results.append(result)
        
        return results


def create_style_transfer_model():
    """Factory function to create style transfer pipeline"""
    return StyleTransferPipeline()


if __name__ == "__main__":
    # Example usage
    pipeline = StyleTransferPipeline()
    
    # Example with AdaIN (fast)
    # pipeline.transfer_style_adain('content.jpg', 'style.jpg', 'output_adain.jpg')
    
    # Example with optimization (higher quality)
    # pipeline.transfer_style_optimization('content.jpg', 'style.jpg', 'output_opt.jpg')
