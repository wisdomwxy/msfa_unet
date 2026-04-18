import torch
import torch.nn as nn
from timm.models.mobilenetv3 import mobilenetv3_large_100, mobilenetv3_small_100
import torch.nn.functional as F

class MobileNetV3(nn.Module):
    def __init__(self, pretrained=False, variant='large'):
        super(MobileNetV3, self).__init__()
        if variant == 'large':
            base_model = mobilenetv3_large_100(pretrained=pretrained)
        else:  # small
            base_model = mobilenetv3_small_100(pretrained=pretrained)

        if hasattr(base_model, 'act1'):
            self.stage0 = nn.Sequential(
                base_model.conv_stem,
                base_model.bn1,
                base_model.act1
            )
        elif hasattr(base_model, 'act'):
            self.stage0 = nn.Sequential(
                base_model.conv_stem,
                base_model.bn1,
                base_model.act
            )
        else:
            # Adapted to the structure of the new version of the timm library
            self.stage0 = nn.Sequential(
                base_model.conv_stem,
                base_model.bn1,
                nn.ReLU(inplace=True) 
            )
        
        if not hasattr(base_model, 'blocks'):
            raise AttributeError("The MobileNetV3 model from timm does not have 'blocks' attribute. Please check the model structure.")
        

        
  
        total_blocks = len(base_model.blocks)
        
        if variant == 'large':
            if total_blocks == 7:
                self.stage1 = nn.Sequential(*base_model.blocks[0:1])    
                self.stage2 = nn.Sequential(*base_model.blocks[1:3])    
                self.stage3 = nn.Sequential(*base_model.blocks[3:5])    
                self.stage4 = nn.Sequential(*base_model.blocks[5:7])    
            else:
                stage1_end = max(1, total_blocks // 7)
                stage2_end = max(2, total_blocks * 2 // 7)
                stage3_end = max(4, total_blocks * 4 // 7)
                self.stage1 = nn.Sequential(*base_model.blocks[0:stage1_end])
                self.stage2 = nn.Sequential(*base_model.blocks[stage1_end:stage2_end])
                self.stage3 = nn.Sequential(*base_model.blocks[stage2_end:stage3_end])
                self.stage4 = nn.Sequential(*base_model.blocks[stage3_end:])
        else:
            if total_blocks == 6:
                self.stage1 = nn.Sequential(*base_model.blocks[0:1])    
                self.stage2 = nn.Sequential(*base_model.blocks[1:2])    
                self.stage3 = nn.Sequential(*base_model.blocks[2:4])    
                self.stage4 = nn.Sequential(*base_model.blocks[4:6])    
            else:
                stage1_end = max(1, total_blocks // 6)
                stage2_end = max(2, total_blocks * 2 // 6)
                stage3_end = max(4, total_blocks * 4 // 6)
                self.stage1 = nn.Sequential(*base_model.blocks[0:stage1_end])
                self.stage2 = nn.Sequential(*base_model.blocks[stage1_end:stage2_end])
                self.stage3 = nn.Sequential(*base_model.blocks[stage2_end:stage3_end])
                self.stage4 = nn.Sequential(*base_model.blocks[stage3_end:])
        
        self.variant = variant

    def forward(self, x):
        input_shape = x.shape[-2:]  
        
        target_sizes = [
            (torch.div(input_shape[0], 2, rounding_mode='floor'), torch.div(input_shape[1], 2, rounding_mode='floor')),     # feat1: 1/2
            (torch.div(input_shape[0], 4, rounding_mode='floor'), torch.div(input_shape[1], 4, rounding_mode='floor')),     # feat2: 1/4
            (torch.div(input_shape[0], 8, rounding_mode='floor'), torch.div(input_shape[1], 8, rounding_mode='floor')),     # feat3: 1/8
            (torch.div(input_shape[0], 16, rounding_mode='floor'), torch.div(input_shape[1], 16, rounding_mode='floor')),   # feat4: 1/16
            (torch.div(input_shape[0], 32, rounding_mode='floor'), torch.div(input_shape[1], 32, rounding_mode='floor'))    # feat5: 1/32
        ]
        
        feat1 = self.stage0(x)
        feat1 = F.interpolate(feat1, size=target_sizes[0], mode='bilinear', align_corners=False)
        
        feat2 = self.stage1(feat1)
        feat2 = F.interpolate(feat2, size=target_sizes[1], mode='bilinear', align_corners=False)
        
        feat3 = self.stage2(feat2)
        feat3 = F.interpolate(feat3, size=target_sizes[2], mode='bilinear', align_corners=False)
        
        feat4 = self.stage3(feat3)
        feat4 = F.interpolate(feat4, size=target_sizes[3], mode='bilinear', align_corners=False)
        
        feat5 = self.stage4(feat4)
        feat5 = F.interpolate(feat5, size=target_sizes[4], mode='bilinear', align_corners=False)
        

        
        return [feat1, feat2, feat3, feat4, feat5]

def mobilenetv3_large(pretrained=False):
    return MobileNetV3(pretrained=pretrained, variant='large')

def mobilenetv3_small(pretrained=False):
    return MobileNetV3(pretrained=pretrained, variant='small') 