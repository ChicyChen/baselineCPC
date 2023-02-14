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

    model = CPC_1layer_2d_static_M0()

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
        loss = model(input_seq)
        print(loss.size())
        sys.exit("test end.") 


# baseline 2D
class CPC_1layer_2d_static_M0(nn.Module):
    def __init__(self, code_size=512, top_size=512, pred_step=3, nsub=3, useout=True, seeall=False, loss_mode=0):
        super().__init__()
        torch.cuda.manual_seed(233)

        self.code_size = code_size
        self.top_size = top_size
        self.pred_step = pred_step
        self.nsub = nsub
        self.useout = useout
        self.seeall = seeall
        self.mask = None
        self.criterion = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.loss_mode = loss_mode

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
        # (X, 512, 7, 7) -> (X, 512, 7, 7)
        self.auto_agressive = ConvGRU(in_channels=512, hidden_channels=512, kernel_size=(1,1), num_layers=1,
            batch_first=True, bias=True, return_all_layers=False
        )
        # (X, 512, 7, 7) -> (X, 512, 7, 7)
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
        c_true, hidden = self.auto_agressive(feature)
        c_true = c_true[-1] # B, N, self.top_size, 7, 7

        if self.seeall:
            pred = self.latent_pred(c_true.view(B*N, self.top_size, 7, 7)).view(B, N, self.code_size, 7, 7)
            pred = pred[:, N-self.pred_step:, :].contiguous()
            c_pred, _ = self.auto_agressive(pred)
            c_pred = c_pred[-1]
        else:
            output, hidden = self.auto_agressive(feature[:, 0:N-self.pred_step, :].contiguous())
            output = output[-1][:,-1,:]
            # hidden[-1].size(): B, 512, 7, 7

            pred = []
            c_pred = []
            for i in range(self.pred_step):
                # sequentially pred future
                p_tmp = self.latent_pred(output)
                # p_tmp.size(): B, 512, 7, 7
                pred.append(p_tmp)
                output, hidden = self.auto_agressive(p_tmp.unsqueeze(1), hidden)
                output = output[-1][:,-1,:]
                c_pred.append(output)
            pred = torch.stack(pred, 1)  # B, pred_step, 512, 7, 7
            c_pred = torch.stack(c_pred, 1)

        N_sub = self.nsub  # cobtrol number of negative pairs
        feature_sub = feature[:, N-N_sub:, :].contiguous() # real zt for contrast
        c_sub = c_true[:, N-N_sub:, :].contiguous() # real ct for contrast

        feature_real = feature[:, N-self.pred_step:, :].contiguous() # real zt for mse: B, pred_step, 512, 7, 7
        c_real = c_true[:, N-self.pred_step:, :].contiguous() # real ct for mse: B, pred_step, 512, 7, 7

        similarity = torch.matmul(
            pred.permute(0,1,3,4,2).contiguous().view(B*self.pred_step*7*7, self.code_size), 
            feature_sub.permute(0,1,3,4,2).contiguous().view(B*N_sub*7*7, self.code_size).transpose(0, 1))
        c_similarity = torch.matmul(
            c_pred.permute(0,1,3,4,2).contiguous().view(B*self.pred_step*7*7, self.code_size), 
            c_sub.permute(0,1,3,4,2).contiguous().view(B*N_sub*7*7, self.code_size).transpose(0, 1))
        
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
        
        loss_zc = self.criterion(similarity, self.mask)
        loss_cc = self.criterion(c_similarity, self.mask)
        loss_zmse = self.mse(feature_real, pred)
        loss_cmse = self.mse(c_real, c_pred)

        if self.loss_mode == 0:
            loss = loss_zc
        elif self.loss_mode == 1:
            loss = loss_cc
        elif self.loss_mode == 2:
            loss = loss_zmse
        elif self.loss_mode == 3:
            loss = loss_cmse
        elif self.loss_mode == 4:
            loss = loss = loss_zc + loss_cc + loss_zmse + loss_cmse

        return loss

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def reset_mask(self):
        self.mask = None


class action_CPC_1layer_2d_static_M0(nn.Module):
    def __init__(self, code_size=512, top_size=512, class_num=101):
        super().__init__()
        torch.cuda.manual_seed(233)

        self.code_size = code_size
        self.top_size = top_size
        self.class_num = class_num

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

        self.predhead = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.class_num)
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
        # print(feature.size())
        feature = feature.view(B, N, self.code_size, 7, 7)
        context, _ = self.auto_agressive(feature.contiguous()) # B, N, 512, 7, 7
        context = context[-1][:,-1,:] # B, 512, 7, 7
        # print(context.size())
        # sys.exit("test end.") 
        output = self.predhead(context)
        # print(output.size())
        # sys.exit("test end.") 
        
        return output

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)



if __name__ == '__main__':
    main()