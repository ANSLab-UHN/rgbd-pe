import torch
import torch.nn as nn
from torch.nn import functional as F
from backbone import *
from config import cfg
import os.path as osp

model_urls = {
    'MobileNetV2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'ResNet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'ResNet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'ResNet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'ResNet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ResNet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'ResNext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

BACKBONE_DICT = {
    'LPRES':LpNetResConcat,
    'LPSKI':LpNetSkiConcat,
    'LPWO':LpNetWoConcat,
    'catNET': catNet
    }

def soft_argmax(heatmaps, joint_num):                   # torch.Size([64, 576, 32, 32])
    # Reshaping keeps the original order of things. So now we go from 576 channels of dims 32x32 to 18 channels of len(32768).
        # The -1 keeps means torch will automatically calculate what the first dimension (batch size) will be. In this case, it's kept as the same cause the rest adds up.
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim*cfg.output_shape[0]*cfg.output_shape[1]))         # torch.Size([64, 18, 32768])
    heatmaps = F.softmax(heatmaps, 2)                                                                           # torch.Size([64, 18, 32768])
    heatmaps = heatmaps.reshape((-1, joint_num, cfg.depth_dim, cfg.output_shape[0], cfg.output_shape[1]))       # torch.Size([64, 18, 32, 32, 32])

    accu_x = heatmaps.sum(dim=(2,3))    # -> have dim = 4 left
    accu_y = heatmaps.sum(dim=(2,4))    # -> have dim = 3 left
    accu_z = heatmaps.sum(dim=(3,4))    # -> have dim = 2 left

    accu_x = accu_x * torch.arange(1,cfg.output_shape[1]+1, device = 'cuda')
    accu_y = accu_y * torch.arange(1,cfg.output_shape[0]+1, device = 'cuda')
    accu_z = accu_z * torch.arange(1,cfg.depth_dim+1, device = 'cuda')

    # at this point you end up with [64, 18] each
    accu_x = accu_x.sum(dim=2, keepdim=True) -1
    accu_y = accu_y.sum(dim=2, keepdim=True) -1
    accu_z = accu_z.sum(dim=2, keepdim=True) -1

    # then they get concatenated and you get [64, 18, 3]. the three numbers in the last dimension are the x/y/z coords of each joint (18) for each batch
    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)          # torch.Size([64, 18, 3])

    return coord_out

class CustomNet(nn.Module):
    def __init__(self, backbone, joint_num):
        super(CustomNet, self).__init__()
        self.backbone = backbone
        self.joint_num = joint_num
        #self.conv = nn.Conv2d(in_channels=1152, out_channels=576, kernel_size=1, stride=1, padding=0)

    def forward(self, input_img, target=None):
        fm = self.backbone(input_img)           # torch.Size([32, 576, 32, 32])
        #fm = self.conv(fm)

        coord = soft_argmax(fm, self.joint_num)

        if target is None:
            return coord
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']

            ## coordinate loss
            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (loss_coord[:,:,0] + loss_coord[:,:,1] + loss_coord[:,:,2] * target_have_depth)/3.
            return loss_coord

def get_pose_net(backbone_str, is_train, joint_num):
    INPUT_SIZE = cfg.input_shape
    EMBEDDING_SIZE = cfg.embedding_size # feature dimension
    WIDTH_MULTIPLIER = cfg.width_multiplier

    assert INPUT_SIZE == (256, 256)

    print("=" * 60)
    print("{} BackBone Generated".format(backbone_str))
    print("=" * 60)
    model = CustomNet(BACKBONE_DICT[backbone_str](input_size = INPUT_SIZE, joint_num = joint_num, embedding_size = EMBEDDING_SIZE, width_mult = WIDTH_MULTIPLIER), joint_num)
    if is_train == True:
        model.backbone.init_weights()
    return model
