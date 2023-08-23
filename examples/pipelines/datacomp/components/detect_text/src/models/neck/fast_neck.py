import torch.nn as nn
from models.utils.nas_utils import set_layer_from_config
import torch.nn.functional as F
import torch
import json


class FASTNeck(nn.Module):
    def __init__(self, reduce_layer1, reduce_layer2, reduce_layer3, reduce_layer4):
        super(FASTNeck, self).__init__()
        self.reduce_layer1 = reduce_layer1
        self.reduce_layer2 = reduce_layer2
        self.reduce_layer3 = reduce_layer3
        self.reduce_layer4 = reduce_layer4

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode="bilinear")

    def forward(self, x):
        f1, f2, f3, f4 = x
        f1 = self.reduce_layer1(f1)
        f2 = self.reduce_layer2(f2)
        f3 = self.reduce_layer3(f3)
        f4 = self.reduce_layer4(f4)

        f2 = self._upsample(f2, f1)
        f3 = self._upsample(f3, f1)
        f4 = self._upsample(f4, f1)
        f = torch.cat((f1, f2, f3, f4), 1)
        return f

    @staticmethod
    def build_from_config(config):
        reduce_layer1 = set_layer_from_config(config["reduce_layer1"])
        reduce_layer2 = set_layer_from_config(config["reduce_layer2"])
        reduce_layer3 = set_layer_from_config(config["reduce_layer3"])
        reduce_layer4 = set_layer_from_config(config["reduce_layer4"])
        return FASTNeck(reduce_layer1, reduce_layer2, reduce_layer3, reduce_layer4)


def fast_neck(config, **kwargs):
    neck_config = json.load(open(config, "r"))["neck"]
    neck = FASTNeck.build_from_config(neck_config, **kwargs)
    return neck
