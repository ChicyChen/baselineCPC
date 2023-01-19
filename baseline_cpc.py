import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class baseline_CPC(nn.Module):
    # pre_frame_num = 5, pred_step = 3, in total 8 frames
    def __init__(self, code_size=128, pred_step=3):
        super().__init__()

        self.code_size = code_size
        self.pred_step = pred_step
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

        self._initialize_weights(self.gar)
        self._initialize_weights(self.pred)

    def forward(self, block):
        # block: [B, N, C, H, W]
        (B, N, C, H, W) = block.shape
        # print(B)

        block = block.view(B*N, C, H, W)
        feature = self.genc(block)  # [B*N, code_size]
        feature = feature.view(B, N, self.code_size)
        # print(feature.size())
        _, hidden = self.gar(feature[:, 0:N-self.pred_step, :].contiguous())
        # print(hidden.size())
        # hidden = hidden[:, -1, :]
        # print(hidden.size())

        pred = []
        for i in range(self.pred_step):
            # sequentially pred future
            p_tmp = self.pred(hidden)
            p_tmp = p_tmp.squeeze(0)
            pred.append(p_tmp)
            _, hidden = self.gar(p_tmp.unsqueeze(1), hidden)
            # hidden = hidden[:, -1, :]
        pred = torch.stack(pred, 1)  # B, pred_step, code_size

        # print(pred.size())

        ### Get similarity score ###
        # pred: [B, pred_step, code_size]
        # feature: [B, N, code_size]
        # feature_sub = [B, N_sub, code_size]
        N_sub = pred_step # cobtrol number of negative pairs
        feature_sub = feature[:, N-N_sub:, :]
        similarity = torch.matmul(pred.view(B*self.pred_step, self.code_size), feature_sub.view(
            B*N, self.code_size).transpose(0, 1)).view(B, self.pred_step, B, N_sub)

        if self.mask is None:
            mask = torch.zeros((B, self.pred_step, B, N_sub),
                               dtype=torch.int8, requires_grad=False)
            # mask = mask.detach().cuda() # TODO GPU
            for j in range(B):
                mask[j, torch.arange(self.pred_step), j, torch.arange(
                    N_sub-self.pred_step, N_sub)] = 1  # pos
            self.mask = mask

        return [similarity, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def reset_mask(self):
        self.mask = None
