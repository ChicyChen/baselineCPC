import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPC_2layer_1d_static(nn.Module):
    
    def __init__(self, code_size=[256,256], top_size=256, pred_step=3, nsub=3):
        super().__init__()
        torch.cuda.manual_seed(233)

        self.code_size = code_size
        self.top_size = top_size
        self.pred_step = pred_step
        self.nsub = nsub
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
            nn.Linear(256, self.code_size[0])
        )

        self.gar1 = nn.GRU(self.code_size[0], self.code_size[1], batch_first=True)
        self.gar2 = nn.GRU(self.code_size[1], self.top_size, batch_first=True)

        self.pred2 = nn.Linear(self.top_size, self.code_size[1])
        self.pred1 = nn.Linear(self.code_size[1], self.code_size[0])

        self._initialize_weights(self.gar1)
        self._initialize_weights(self.gar2)
        self._initialize_weights(self.pred1)
        self._initialize_weights(self.pred2)

    def forward(self, block):
        # block: [B, N, C, H, W]
        (B, N, C, H, W) = block.shape

        block = block.view(B*N, C, H, W)
        feature = self.genc(block)  # [B*N, code_size]
        feature = feature.view(B, N, self.code_size[0])
        out, hidden = self.gar1(feature[:, 0:N-self.pred_step, :].contiguous())
        _, hidden2 = self.gar2(out)

        pred = []
        out_middle = []
        pred2 = []
        for i in range(self.pred_step):
            # sequentially pred future
            p_tmp = self.pred1(hidden.squeeze(0))
            pred.append(p_tmp)
            o_tmp, hidden = self.gar1(p_tmp.unsqueeze(1), hidden)
            o_tmp = o_tmp.squeeze(1)
            out_middle.append(o_tmp)

            p_tmp2 = self.pred2(hidden2)
            p_tmp2 = p_tmp2.squeeze(0)
            pred2.append(p_tmp2)
            _, hidden2 = self.gar2(p_tmp2.unsqueeze(1), hidden2)

        pred = torch.stack(pred, 1)  # B, pred_step, code_size0
        out_middle = torch.stack(out_middle, 1) # B, pred_step, code_size1
        pred2 = torch.stack(pred2, 1)  # B, pred_step, code_size1

        ### Get similarity score ###
        N_sub = self.nsub  # cobtrol number of negative pairs

        # pred: [B, pred_step, code_size0]
        # feature: [B, N, code_size0]
        # feature_sub = [B, N_sub, code_size0]
        feature_sub = feature[:, N-N_sub:, :].contiguous()
        similarity = torch.matmul(pred.view(B*self.pred_step, self.code_size[0]), feature_sub.view(
            B*N_sub, self.code_size[0]).transpose(0, 1)).view(B, self.pred_step, B, N_sub)

        if self.mask is None:
            mask = torch.zeros((B, self.pred_step, B, N_sub),
                               dtype=torch.int8, requires_grad=False)
            mask = mask.detach().cuda()
            for j in range(B):
                mask[j, torch.arange(self.pred_step), j, torch.arange(
                    N_sub-self.pred_step, N_sub)] = 1  # pos
            self.mask = mask

        # pred2: [B, pred_step, code_size1]
        # out: [B, N-pred_step, code_size1]
        # out_middle: [B, pred_step, code_size1]
        # feature_sub2 = [B, N_sub, code_size1]
        
        if N_sub > self.pred_step:
            feature_sub2 = torch.cat((out[:, N_sub-self.pred_step:, :], out_middle),1)
        else:
            feature_sub2 = out_middle[:, self.pred_step-N_sub:, :]
        # print(pred2.size(), feature_sub2.size())
        similarity2 = torch.matmul(pred2.view(B*self.pred_step, self.code_size[1]), feature_sub2.view(
            B*N_sub, self.code_size[1]).transpose(0, 1)).view(B, self.pred_step, B, N_sub)

        return [similarity, similarity2, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def reset_mask(self):
        self.mask = None


class motion_CPC_2layer_1d_static(nn.Module):
    
    def __init__(self, code_size=[256,256], top_size=256):
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
            nn.Linear(256, self.code_size[0])
        )

        self.gar1 = nn.GRU(self.code_size[0], self.code_size[1], batch_first=True)
        self.gar2 = nn.GRU(self.code_size[1], self.top_size, batch_first=True)

        self.pred2 = nn.Linear(self.top_size, self.code_size[1])
        self.pred1 = nn.Linear(self.code_size[1], self.code_size[0])

        # self.predmotion = nn.Sequential(
        #     nn.Linear(256, 64),
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

        self._initialize_weights(self.gar1)
        self._initialize_weights(self.gar2)
        self._initialize_weights(self.pred1)
        self._initialize_weights(self.pred2)

    def forward(self, block):
        # block: [B, N, C, H, W]
        (B, N, C, H, W) = block.shape

        block = block.view(B*N, C, H, W)
        feature = self.genc(block)  # [B*N, code_size]
        feature = feature.view(B, N, self.code_size[0])
        out, _ = self.gar1(feature.contiguous())
        context, _ = self.gar2(out)
        context = context[:, -1, :]
        output = self.predmotion(context).view(B, 6)

        return [output, context]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


class digit_CPC_2layer_1d_static(nn.Module):
    
    def __init__(self, code_size=[256,256], top_size=256):
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
            nn.Linear(256, self.code_size[0])
        )

        self.gar1 = nn.GRU(self.code_size[0], self.code_size[1], batch_first=True)
        self.gar2 = nn.GRU(self.code_size[1], self.top_size, batch_first=True)

        self.pred2 = nn.Linear(self.top_size, self.code_size[1])
        self.pred1 = nn.Linear(self.code_size[1], self.code_size[0])

        # self.preddigit = nn.Sequential(
        #     nn.Linear(256, 64),
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

        self._initialize_weights(self.gar1)
        self._initialize_weights(self.gar2)
        self._initialize_weights(self.pred1)
        self._initialize_weights(self.pred2)

    def forward(self, block):
        # block: [B, N, C, H, W]
        (B, N, C, H, W) = block.shape

        block = block.view(B*N, C, H, W)
        feature = self.genc(block)  # [B*N, code_size]
        feature = feature.view(B, N, self.code_size[0])
        out, _ = self.gar1(feature.contiguous())
        context, _ = self.gar2(out)
        context = context[:, -1, :]
        output = self.preddigit(context).view(B, 10)

        return [output, context]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)