import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_UCF101 import *

from backbone import *
from auto_aggressive import *



def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str('0,1,2,3')
    global cuda
    cuda = torch.device('cuda')

    model = CPC_1layer_2d_static()

    model = nn.DataParallel(model)
    model = model.to(cuda)

    transform = transforms.Compose([
        RandomHorizontalFlip(consistent=True),
        RandomCrop(size=224, consistent=True),
        Scale(size=(224,224)),
        RandomGray(consistent=False, p=0.5),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
        ToTensor(),
        Normalize()
    ])
    train_loader = get_data_ucf(transform, 'train', batch_size=8, dim=240)

    for _, input_seq in enumerate(train_loader):
        input_seq = input_seq.to(cuda)
        out = model(input_seq)
        # print(out.size())


class CPC_1layer_2d_static(nn.Module):
    def __init__(self, code_size=512, top_size=512, pred_step=3, nsub=3):
        super().__init__()
        torch.cuda.manual_seed(233)

        self.code_size = code_size
        self.top_size = top_size
        self.pred_step = pred_step
        self.nsub = nsub
        self.mask = None

        self.net_pre = nn.Sequential(
            # (X, 3, 224, 224) -> (X, 64, 112, 112)
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (X, 64, 112, 112) -> (X, 64, 56, 56)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # (X, 64, 56, 56) -> (X, 512, 7, 7)
        self.backbone = nn.Sequential(
            ResNetLayer(BasicBlock, 2, inplanes=64, planes=64, stride=1),
            ResNetLayer(BasicBlock, 2, inplanes=64, planes=128, stride=2),
            ResNetLayer(BasicBlock, 2, inplanes=128, planes=256, stride=2),
            ResNetLayer(BasicBlock, 2, inplanes=256, planes=512, stride=2)
            # ResNetLayer(BasicBlock, 2, inplanes=256, planes=256, stride=2)   # In DPC, final layer does not have a larger channel number
        )
        # (X, 512, 7, 7) -> (X, 512, ?, ?)
        self.auto_agressive = ConvGRU(in_channels=512, hidden_channels=512, kernel_size=(1,1), num_layers=1,
            batch_first=True, bias=True, return_all_layers=False
        )
        # (X, 512, ?, ?) -> (X, 512, ?, ?)
        self.latent_pred = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0)
        )

        # Initialize weights
        # self._initialize_weights(self.net_pre)
        self._initialize_weights(self.auto_agressive)
        self._initialize_weights(self.latent_pred)
    
    def forward(self, block):
        (B, N, C, H, W) = block.shape
        block = block.view(B*N, C, H, W)
        feature = self.net_pre(block) # B*N, 64, 56, 56
        feature = self.backbone(feature) # B*N, 512, 7, 7
        feature = feature.view(B, N, self.code_size, 7, 7)
        _, hidden = self.auto_agressive(feature[:, 0:N-self.pred_step, :].contiguous())
        # hidden[-1].size(): B, 512, 7, 7

        pred = []
        for i in range(self.pred_step):
            # sequentially pred future
            p_tmp = self.latent_pred(hidden[-1])
            # p_tmp.size(): B, 512, 7, 7
            pred.append(p_tmp)
            _, hidden = self.auto_agressive(p_tmp.unsqueeze(1), hidden)
        pred = torch.stack(pred, 1)  # B, pred_step, 512, 7, 7

        N_sub = self.nsub  # cobtrol number of negative pairs
        feature_sub = feature[:, N-N_sub:, :].contiguous()
        similarity = torch.matmul(
            pred.permute(0,1,3,4,2).contiguous().view(B*self.pred_step*7*7, self.code_size), 
            feature_sub.permute(0,1,3,4,2).contiguous().view(B*N_sub*7*7, self.code_size).transpose(0, 1))
        
        if self.mask is None:
            mask = torch.zeros((B, self.pred_step, 7*7, B, N_sub, 7*7), dtype=torch.int8, requires_grad=False).detach().cuda()
            # print(mask.size())
            mask = mask.detach().cuda()
            for j in range(B):
                for i in range(self.pred_step):
                    mask[j, i, torch.arange(7*7), j, N_sub-self.pred_step+i, torch.arange(7*7)] = 1  # pos
            mask = mask.view(B*self.pred_step*7*7, B*N_sub*7*7)
            self.mask = mask.to(int).argmax(dim=1)
            # print(torch.sum(mask))
            # sys.exit("test end.") 
            # print(similarity.size(), self.mask.size())
        
        return [similarity, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def reset_mask(self):
        self.mask = None


class CPC_1layer_1d_static(nn.Module):
    
    def __init__(self, code_size=256, top_size=256, pred_step=3, nsub=3):
        super().__init__()
        torch.cuda.manual_seed(233)

        self.code_size = code_size
        self.top_size = top_size
        self.pred_step = pred_step
        self.nsub = nsub
        self.mask = None

        self.genc = nn.Sequential(
            # (X, 3, 128, 128) -> (X, 16, 64, 64)
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # (X, 16, 64, 64) -> (X, 32, 32, 32)
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (X, 32, 32, 32) -> (X, 64, 16, 16)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (X, 64, 16, 16) -> (X, 128, 8, 8)
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*8*8, 64*4*4),
            nn.BatchNorm1d(64*4*4),
            nn.ReLU(),
            nn.Linear(64*4*4, self.code_size)
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

        self._initialize_weights(self.gar)
        self._initialize_weights(self.pred)

    def forward(self, block):
        # block: [B, N, C, H, W]
        (B, N, C, H, W) = block.shape
        # print(B, N, C, H, W)

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
            p_tmp = self.pred(hidden.squeeze(0))
            pred.append(p_tmp)
            _, hidden = self.gar(p_tmp.unsqueeze(1), hidden)
            # hidden = hidden[:, -1, :]
        pred = torch.stack(pred, 1)  # B, pred_step, code_size

        # print(pred.size())

        ### Get similarity score ###
        # pred: [B, pred_step, code_size]
        # feature: [B, N, code_size]
        # feature_sub = [B, N_sub, code_size]
        N_sub = self.nsub  # cobtrol number of negative pairs
        feature_sub = feature[:, N-N_sub:, :].contiguous()
        similarity = torch.matmul(pred.view(B*self.pred_step, self.code_size), feature_sub.view(
            B*N_sub, self.code_size).transpose(0, 1))

        if self.mask is None:
            mask = torch.zeros((B, self.pred_step, B, N_sub),
                               dtype=torch.int8, requires_grad=False)
            mask = mask.detach().cuda()
            for j in range(B):
                mask[j, torch.arange(self.pred_step), j, torch.arange(
                    N_sub-self.pred_step, N_sub)] = 1  # pos
            mask_flattened = mask.view(B*self.pred_step, B*self.nsub)
            self.mask = mask_flattened.to(int).argmax(dim=1)

        # print(score.size(), self.mask.size())

        return [similarity, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def reset_mask(self):
        self.mask = None




class action_CPC_1layer_1d_static(nn.Module):
    
    def __init__(self, code_size=256, top_size=256, class_num=101):
        super().__init__()
        torch.cuda.manual_seed(233)

        self.code_size = code_size
        self.top_size = top_size
        self.class_num = class_num

        self.genc = nn.Sequential(
            # (X, 3, 128, 128) -> (X, 16, 64, 64)
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # (X, 16, 64, 64) -> (X, 32, 32, 32)
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (X, 32, 32, 32) -> (X, 64, 16, 16)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (X, 64, 16, 16) -> (X, 128, 8, 8)
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*8*8, 64*4*4),
            nn.BatchNorm1d(64*4*4),
            nn.ReLU(),
            nn.Linear(64*4*4, self.code_size)
            # nn.BatchNorm1d(self.code_size),
            # nn.ReLU()
        )

        self.gar = nn.GRU(self.code_size, self.top_size, batch_first=True)

        self.pred = nn.Linear(self.top_size, self.code_size)

        self.predhead = nn.Sequential(
            nn.BatchNorm1d(self.top_size),
            nn.Linear(self.top_size, self.class_num)
        )

        self._initialize_weights(self.gar)


    def forward(self, block):
        # block: [B, N, C, H, W]
        (B, N, C, H, W) = block.shape
        # print(B, N, C, H, W)

        block = block.view(B*N, C, H, W)
        feature = self.genc(block)  # [B*N, code_size]
        feature = feature.view(B, N, self.code_size)
        context, _ = self.gar(feature.contiguous())
        context = context[:, -1, :]
        output = self.predhead(context).view(B, self.class_num)

        return [output, context]


    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)




if __name__ == '__main__':
    main()
    