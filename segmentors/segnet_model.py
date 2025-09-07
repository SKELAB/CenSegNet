import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

class SegNet(nn.Module):
    def __init__(self, config):
        super(SegNet, self).__init__()
        
        if isinstance(config, dict):
            self.inch = config.get('inch', 3)
            self.outch = config.get('outch', 1)
            self.batchNorm_momentum = config.get('batchNorm_momentum', 0.1)
            self.encoder_filters = config.get('encoder_filters', [64, 128, 256, 512])
            self.decoder_filters = config.get('decoder_filters', [512, 256, 128, 64])
            self.kernel_sizes = config.get('kernel_sizes', [3, 3, 3, 3])
            self.padding = config.get('padding', [1, 1, 1, 1])
        else:
            self.inch = config if isinstance(config, int) else 3
            self.outch = 1
            self.batchNorm_momentum = 0.1
            self.encoder_filters = [64, 128, 256, 512]
            self.decoder_filters = [512, 256, 128, 64]
            self.kernel_sizes = [3, 3, 3, 3]
            self.padding = [1, 1, 1, 1]
        
        self.encoder_stages = nn.ModuleList()
        in_channels = self.inch
        
        for stage_idx, out_channels in enumerate(self.encoder_filters):
            stage = nn.ModuleList()
            stage.append(nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=self.kernel_sizes[min(stage_idx, len(self.kernel_sizes)-1)], 
                                   padding=self.padding[min(stage_idx, len(self.padding)-1)]))
            stage.append(nn.BatchNorm2d(out_channels, momentum=self.batchNorm_momentum))
            stage.append(nn.ReLU(inplace=True))
            
            stage.append(nn.Conv2d(out_channels, out_channels, 
                                   kernel_size=self.kernel_sizes[min(stage_idx, len(self.kernel_sizes)-1)], 
                                   padding=self.padding[min(stage_idx, len(self.padding)-1)]))
            stage.append(nn.BatchNorm2d(out_channels, momentum=self.batchNorm_momentum))
            stage.append(nn.ReLU(inplace=True))
            if out_channels >= 256:
                stage.append(nn.Conv2d(out_channels, out_channels, 
                                       kernel_size=self.kernel_sizes[min(stage_idx, len(self.kernel_sizes)-1)], 
                                       padding=self.padding[min(stage_idx, len(self.padding)-1)]))
                stage.append(nn.BatchNorm2d(out_channels, momentum=self.batchNorm_momentum))
                stage.append(nn.ReLU(inplace=True))
            
            self.encoder_stages.append(stage)
            in_channels = out_channels
        
        self.decoder_stages = nn.ModuleList()
        for stage_idx, out_channels in enumerate(self.decoder_filters):
            stage = nn.ModuleList()
            
            encoder_stage = self.encoder_stages[len(self.encoder_stages) - 1 - stage_idx]
            conv_count = len([l for l in encoder_stage if isinstance(l, nn.Conv2d)])
            
            in_channels = self.encoder_filters[len(self.encoder_filters) - 1 - stage_idx]
            
            for i in range(conv_count):
                if i == conv_count - 1:  
                    stage.append(nn.Conv2d(in_channels, out_channels, 
                                          kernel_size=self.kernel_sizes[min(stage_idx, len(self.kernel_sizes)-1)], 
                                          padding=self.padding[min(stage_idx, len(self.padding)-1)]))
                    stage.append(nn.BatchNorm2d(out_channels, momentum=self.batchNorm_momentum))
                    stage.append(nn.ReLU(inplace=True))
                    in_channels = out_channels
                else:
                    stage.append(nn.Conv2d(in_channels, in_channels, 
                                          kernel_size=self.kernel_sizes[min(stage_idx, len(self.kernel_sizes)-1)], 
                                          padding=self.padding[min(stage_idx, len(self.padding)-1)]))
                    stage.append(nn.BatchNorm2d(in_channels, momentum=self.batchNorm_momentum))
                    stage.append(nn.ReLU(inplace=True))
            
            self.decoder_stages.append(stage)
        
        self.final_conv = nn.Conv2d(self.decoder_filters[-1], self.outch, 
                                    kernel_size=self.kernel_sizes[-1], 
                                    padding=self.padding[-1])
    
    def forward(self, x):
        encoder_features = []
        indices = []
        sizes = []
        
        for stage in self.encoder_stages:
            for layer in stage:
                x = layer(x)
            
            sizes.append(x.size())
            encoder_features.append(x)
            x, idx = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
            indices.append(idx)
        
        for i, stage in enumerate(self.decoder_stages):
            x = F.max_unpool2d(x, indices[-(i+1)], kernel_size=2, stride=2, output_size=sizes[-(i+1)])
            
            for layer in stage:
                x = layer(x)
        
        x = self.final_conv(x)
        
        return x

def load_model_from_config(config_path: str) -> SegNet:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return SegNet(config)

