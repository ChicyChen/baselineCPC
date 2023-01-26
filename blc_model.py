import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class baseline_m_lc(nn.Module):
    
    def __init__(self, code_size=128):
        super().__init__()
        torch.cuda.manual_seed(233)

        self.code_size = code_size
        self.mask = None

        self.genc = nn.Sequential(
            # (X, 3, 64, 64) -> (X, 16, 32, 32)
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # (X, 16, 32, 32) -> (X, 32, 16, 16)
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (X, 32, 16, 16) -> (X, 64, 8, 8)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (X, 64, 8, 8) -> (X, 64, 4, 4)
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*4*4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.code_size)
        )

        self.gar = nn.GRU(self.code_size, 256, batch_first=True)

        self.pred = nn.Linear(256, self.code_size)

        self.predmotion = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 6),
        )

        self._initialize_weights(self.gar)
        self._initialize_weights(self.pred)
        # self._initialize_weights(self.predmotion)

    def forward(self, block):
        # block: [B, N, C, H, W]
        (B, N, C, H, W) = block.shape
        # print(B, N, C, H, W)

        block = block.view(B*N, C, H, W)
        feature = self.genc(block)  # [B*N, code_size]
        feature = feature.view(B, N, self.code_size)
        context, _ = self.gar(feature.contiguous())
        context = context[:, -1, :]
        # print(context.size())
        output = self.predmotion(context).view(B, 6)
        # print(output.size())

        return [output, context]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def reset_mask(self):
        self.mask = None


class baseline_d_lc(nn.Module):
    
    def __init__(self, code_size=128):
        super().__init__()
        torch.cuda.manual_seed(233)

        self.code_size = code_size
        self.mask = None

        self.genc = nn.Sequential(
            # (X, 3, 64, 64) -> (X, 16, 32, 32)
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # (X, 16, 32, 32) -> (X, 32, 16, 16)
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (X, 32, 16, 16) -> (X, 64, 8, 8)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (X, 64, 8, 8) -> (X, 64, 4, 4)
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*4*4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.code_size)
        )

        self.gar = nn.GRU(self.code_size, 256, batch_first=True)

        self.pred = nn.Linear(256, self.code_size)

        self.preddigit = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 10),
        )

        self._initialize_weights(self.gar)
        self._initialize_weights(self.pred)

    def forward(self, block):
        # block: [B, N, C, H, W]
        (B, N, C, H, W) = block.shape
        # print(B, N, C, H, W)

        block = block.view(B*N, C, H, W)
        feature = self.genc(block)  # [B*N, code_size]
        feature = feature.view(B, N, self.code_size)
        context, _ = self.gar(feature.contiguous())

        # context = torch.mean(context, dim = 1)
        context = context[:, -1, :]
        
        # print(context.size())
        output = self.preddigit(context).view(B, 10)
        # print(output.size())

        return [output, context]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def reset_mask(self):
        self.mask = None