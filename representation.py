import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPC_1layer_1d_static_rep(nn.Module):
    
    def __init__(self, code_size=256, top_size=256, pred_step=3, nsub=3, gt=False):
        super().__init__()
        torch.cuda.manual_seed(233)

        self.code_size = code_size
        self.top_size = top_size
        self.pred_step = pred_step
        self.nsub = nsub
        self.gt = gt
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
            # nn.BatchNorm1d(self.code_size),
            # nn.ReLU()
        )

        self.gar = nn.GRU(self.code_size, self.top_size, batch_first=True)

        # self.pred = nn.Sequential(
        #     nn.Linear(self.top_size, self.code_size),
        #     nn.BatchNorm1d(self.code_size),
        #     nn.ReLU(),
        #     nn.Linear(self.code_size, self.code_size),
            # nn.BatchNorm1d(self.code_size),
            # nn.ReLU()
        # )
        # This does not help for simple data, but make things worse

        self.pred = nn.Linear(self.top_size, self.code_size)
        # may need to change to convolution layers later
        

        self._initialize_weights(self.gar)
        self._initialize_weights(self.pred)

    def forward(self, block):
        # block: [B, N, C, H, W]
        (B, N, C, H, W) = block.shape
        # print(B, N, C, H, W)

        block = block.view(B*N, C, H, W)
        feature = self.genc(block)  # [B*N, code_size]
        feature = feature.view(B, N, self.code_size)
        
        # sequentially process the past
        pi = torch.zeros((B, self.code_size), requires_grad=False)
        pi = pi.detach().cuda()
        hidden = torch.zeros((1, B, self.top_size), requires_grad=False)
        hidden = hidden.detach().cuda()
        for i in range(N - self.pred_step):
            zi = feature[:, i, :]
            si = 0.1*pi + (1-0.1)*zi
            _, hidden = self.gar(si.unsqueeze(1), hidden)
            pi = self.pred(hidden.squeeze(0))
        
        # sequentially pred future
        pred = []
        pred.append(pi)
        for i in range(self.pred_step - 1):
            if self.gt:
                zi = feature[:, i + self.pred_step, :]
                si = 0.1*pi + (1-0.1)*zi
            else:
                si = pi
            _, hidden = self.gar(si.unsqueeze(1), hidden)
            pi = self.pred(hidden.squeeze(0))   # note here hidden state is used for prediction
            pred.append(pi)
            
        pred = torch.stack(pred, 1)  # B, pred_step, code_size

        # print(pred.size())

        ### Get similarity score ###
        # pred: [B, pred_step, code_size]
        # feature: [B, N, code_size]
        # feature_sub = [B, N_sub, code_size]
        N_sub = self.nsub  # cobtrol number of negative pairs
        feature_sub = feature[:, N-N_sub:, :].contiguous()
        similarity = torch.matmul(pred.view(B*self.pred_step, self.code_size), feature_sub.view(
            B*N_sub, self.code_size).transpose(0, 1)).view(B, self.pred_step, B, N_sub)
        # print(similarity.size())

        if self.mask is None:
            mask = torch.zeros((B, self.pred_step, B, N_sub),
                               dtype=torch.int8, requires_grad=False)
            mask = mask.detach().cuda()
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




class motion_CPC_1layer_1d_static_rep(nn.Module):
    
    def __init__(self, code_size=256, top_size=256):
        super().__init__()
        torch.cuda.manual_seed(233)

        self.code_size = code_size
        self.top_size = top_size

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
            # nn.BatchNorm1d(self.code_size),
            # nn.ReLU()
        )

        self.gar = nn.GRU(self.code_size, self.top_size, batch_first=True)

        # self.pred = nn.Sequential(
        #     nn.Linear(self.top_size, self.code_size),
        #     nn.BatchNorm1d(self.code_size),
        #     nn.ReLU(),
        #     nn.Linear(self.code_size, self.code_size),
        #     nn.BatchNorm1d(self.code_size),
        #     nn.ReLU()
        # )

        self.pred = nn.Linear(self.top_size, self.code_size)

        # self.predmotion = nn.Sequential(
        #     nn.Linear(self.top_size, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Linear(64, 16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Linear(16, 6),
        # )

        self.predmotion = nn.Sequential(
            nn.BatchNorm1d(self.top_size),
            nn.Linear(self.top_size, 6)
        )

        self._initialize_weights(self.gar)
        # self._initialize_weights(self.pred)
        # self._initialize_weights(self.predmotion)

    def forward(self, block):
        # block: [B, N, C, H, W]
        (B, N, C, H, W) = block.shape
        # print(B, N, C, H, W)

        block = block.view(B*N, C, H, W)
        feature = self.genc(block)  # [B*N, code_size]
        feature = feature.view(B, N, self.code_size)

        # sequentially process the past
        pi = torch.zeros((B, self.code_size), requires_grad=False)
        pi = pi.detach().cuda()
        hidden = torch.zeros((1, B, self.top_size), requires_grad=False)
        hidden = hidden.detach().cuda()
        for i in range(N):
            zi = feature[:, i, :]
            si = 0.1*pi + (1-0.1)*zi
            context, hidden = self.gar(si.unsqueeze(1), hidden)
            pi = self.pred(hidden.squeeze(0))

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



class digit_CPC_1layer_1d_static_rep(nn.Module):
    
    def __init__(self, code_size=256, top_size=256):
        super().__init__()
        torch.cuda.manual_seed(233)

        self.code_size = code_size
        self.top_size = top_size

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
            # nn.BatchNorm1d(self.code_size),
            # nn.ReLU()
        )

        self.gar = nn.GRU(self.code_size, self.top_size, batch_first=True)

        # self.pred = nn.Sequential(
        #     nn.Linear(self.top_size, self.code_size),
        #     nn.BatchNorm1d(self.code_size),
        #     nn.ReLU(),
        #     nn.Linear(self.code_size, self.code_size),
        #     nn.BatchNorm1d(self.code_size),
        #     nn.ReLU()
        # )

        self.pred = nn.Linear(self.top_size, self.code_size)

        # self.preddigit = nn.Sequential(
        #     nn.Linear(self.top_size, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Linear(64, 16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     nn.Linear(16, 10),
        # )

        self.preddigit = nn.Sequential(
            nn.BatchNorm1d(self.top_size),
            nn.Linear(self.top_size, 10)
        )

        self._initialize_weights(self.gar)
        # self._initialize_weights(self.pred)

    def forward(self, block):
        # block: [B, N, C, H, W]
        (B, N, C, H, W) = block.shape
        # print(B, N, C, H, W)

        block = block.view(B*N, C, H, W)
        feature = self.genc(block)  # [B*N, code_size]
        feature = feature.view(B, N, self.code_size)
        # sequentially process the past
        pi = torch.zeros((B, self.code_size), requires_grad=False)
        pi = pi.detach().cuda()
        hidden = torch.zeros((1, B, self.top_size), requires_grad=False)
        hidden = hidden.detach().cuda()
        for i in range(N):
            zi = feature[:, i, :]
            si = 0.1*pi + (1-0.1)*zi
            context, hidden = self.gar(si.unsqueeze(1), hidden)
            pi = self.pred(hidden.squeeze(0))

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
