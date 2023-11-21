 
import torch
from torch import nn
from torch.nn import functional as f
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            # (batch_size, channel, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),
            
            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),
            
            # (batch_size, 128, height, width) -> (batch_size, 128, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # (batch_size, 128, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256),
            
            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256),
            
            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
            # (batch_size, 256, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),
            
            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),  
            
            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),  
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),  
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_AttentionBlock(512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            nn.GroupNorm(32, 512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            nn.SiLU(),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            
            # (batch_size, 8, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )
    

    def forward(self, x, noise):
        # x: (batch size, channel, height, width)
        # noise: (batch size, 4, height / 8, width / 8)

        for module in self:

            # Padding at downsampling should be asymmetric for conv with strides = 2
            if getattr(module, 'stride', None) == (2, 2):
                # Pad: (left, right, top, bottom)
                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Top + Bottom, Width + Left + Right) 
                # = (Batch_Size, Channel, Height + bottom, Width + right)
                x = f.pad(x, (0, 1, 0, 1))

            x = module(x)

        # (batch_size, 8, height/8, width/8) -> two tensors of shape (batch_size, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        log_variance = torch.clamp(log_variance, -30, 20)

        # (batch_size, 4, height/8, width/8) -> (batch_size, 4, height/8, width/8)
        variance = log_variance.exp()

        stdev = variance.sqrt()

        # Transform Z = N(0, 1) -> X = N(mean, stdev)
        # X = mean + stdev * Z
        x = mean + stdev * noise

        # Scale by a constant
        # Not sure why. From the paper implementation
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        
        # (batch_size, 4, height/8, width/8)
        return x

