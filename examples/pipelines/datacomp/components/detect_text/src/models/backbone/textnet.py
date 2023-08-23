import torch.nn as nn
import json
from models.utils.nas_utils import set_layer_from_config


class TextNet(nn.Module):
    def __init__(self, first_conv, stage1, stage2, stage3, stage4):
        super(TextNet, self).__init__()

        self.first_conv = first_conv
        self.stage1 = nn.ModuleList(stage1)
        self.stage2 = nn.ModuleList(stage2)
        self.stage3 = nn.ModuleList(stage3)
        self.stage4 = nn.ModuleList(stage4)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.first_conv(x)
        output = list()

        for block in self.stage1:
            x = block(x)
        output.append(x)

        for block in self.stage2:
            x = block(x)
        output.append(x)

        for block in self.stage3:
            x = block(x)
        output.append(x)

        for block in self.stage4:
            x = block(x)
        output.append(x)

        return output

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config["first_conv"])
        stage1, stage2, stage3, stage4 = [], [], [], []
        for block_config in config["stage1"]:
            stage1.append(set_layer_from_config(block_config))
        for block_config in config["stage2"]:
            stage2.append(set_layer_from_config(block_config))
        for block_config in config["stage3"]:
            stage3.append(set_layer_from_config(block_config))
        for block_config in config["stage4"]:
            stage4.append(set_layer_from_config(block_config))

        net = TextNet(first_conv, stage1, stage2, stage3, stage4)

        return net


def fast_backbone(config, **kwargs):
    net_config = json.load(open(config, "r"))
    model = TextNet.build_from_config(net_config)
    return model
