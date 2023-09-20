# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch


def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func is None:
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        if len(kernel_size) != 2:
            raise ValueError("invalid kernel size: %s" % kernel_size)
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    if not isinstance(kernel_size, int):
        raise ValueError("kernel size should be either `int` or `tuple`")
    if not (kernel_size % 2 > 0):
        raise ValueError("kernel size should be odd number")
    return kernel_size // 2


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        DepthConvLayer.__name__: DepthConvLayer,
        PoolingLayer.__name__: PoolingLayer,
        IdentityLayer.__name__: IdentityLayer,
        RepVGGLayer.__name__: RepVGGLayer,
        ACBlock.__name__: ACBlock,
        LeftLayer.__name__: LeftLayer,
        AddLayer.__name__: AddLayer,
        ZeroLayer.__name__: ZeroLayer,
        RepConvLayer.__name__: RepConvLayer,
    }

    layer_name = layer_config.pop("name")
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class My2DLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        super(My2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules["bn"] = nn.BatchNorm2d(in_channels)
            else:
                modules["bn"] = nn.BatchNorm2d(out_channels)
        else:
            modules["bn"] = None
        # activation
        modules["act"] = build_activation(self.act_func, self.ops_list[0] != "act")
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # weight
        modules["weight"] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == "weight":
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split("_")

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == "bn":
                return True
            elif op == "weight":
                return False
        raise ValueError("Invalid ops_order: %s" % self.ops_order)

    def weight_op(self):
        raise NotImplementedError

    """ Methods defined in MyModule """

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "use_bn": self.use_bn,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
            "ops_order": self.ops_order,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False


class ConvLayer(My2DLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        has_shuffle=False,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(ConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict["conv"] = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
        )

        return weight_dict

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)


class DepthConvLayer(My2DLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        has_shuffle=False,
        use_bn=True,
        act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle

        super(DepthConvLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict()
        weight_dict["depth_conv"] = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.in_channels,
            bias=False,
        )
        weight_dict["point_conv"] = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            groups=self.groups,
            bias=self.bias,
        )
        return weight_dict

    @staticmethod
    def build_from_config(config):
        return DepthConvLayer(**config)


class PoolingLayer(My2DLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool_type,
        kernel_size=2,
        stride=2,
        use_bn=False,
        act_func=None,
        dropout_rate=0,
        ops_order="weight_bn_act",
    ):
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride

        super(PoolingLayer, self).__init__(
            in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order
        )

    def weight_op(self):
        if self.stride == 1:
            # same padding if `stride == 1`
            padding = get_same_padding(self.kernel_size)
        else:
            padding = 0

        weight_dict = OrderedDict()
        if self.pool_type == "avg":
            weight_dict["pool"] = nn.AvgPool2d(
                self.kernel_size,
                stride=self.stride,
                padding=padding,
                count_include_pad=False,
            )
        elif self.pool_type == "max":
            weight_dict["pool"] = nn.MaxPool2d(
                self.kernel_size, stride=self.stride, padding=padding
            )
        else:
            raise NotImplementedError
        return weight_dict

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)


class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

    @staticmethod
    def build_from_config(config):
        return IdentityLayer()


class ZeroLayer(nn.Module):
    def __init__(self):
        super(ZeroLayer, self).__init__()

    def forward(self, x):
        return x

    @staticmethod
    def build_from_config(config):
        return ZeroLayer()


class LeftLayer(nn.Module):
    def __init__(self):
        super(LeftLayer, self).__init__()

    def forward(self, x):
        return x[0]

    @staticmethod
    def build_from_config(config):
        return LeftLayer()


class AddLayer(nn.Module):
    def __init__(self):
        super(AddLayer, self).__init__()

    def forward(self, x):
        return x[0] + x[1]

    @staticmethod
    def build_from_config(config):
        return AddLayer()


def conv_bn(
    in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1
):
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        deploy=False,
    ):
        super(RepVGGLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.deploy = deploy

        padding = int(((kernel_size - 1) * dilation) / 2)

        self.nonlinearity = nn.ReLU(inplace=True)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )

        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=groups,
            )
            # print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            # print("fast mode")
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        # print("slow mode")
        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            if not isinstance(branch, nn.BatchNorm2d):
                raise ValueError()
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")

    def switch_to_test(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.rbr_reparam.parameters():
            para.detach_()
        self.deploy = True

    def switch_to_train(self):
        if hasattr(self, "rbr_reparam"):
            self.__delattr__("rbr_reparam")
        self.deploy = False

    @staticmethod
    def build_from_config(config):
        return RepVGGLayer(**config)


class ACBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        deploy=False,
    ):
        super(ACBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.deploy = deploy

        padding = int(((kernel_size - 1) * dilation) / 2)
        self.nonlinearity = nn.ReLU(inplace=True)

        if deploy:
            self.fused_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.square_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = 0
            ver_pad_or_crop = (padding, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, padding)

            self.ver_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                stride=stride,
                padding=ver_pad_or_crop,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.hor_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=hor_pad_or_crop,
                dilation=dilation,
                groups=groups,
                bias=False,
            )

            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

    def forward(self, input):
        if hasattr(self, "fused_conv"):
            return self.nonlinearity(self.fused_conv(input))
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = self.ver_conv(input)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv(input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)

            if self.rbr_identity is None:
                id_out = 0
            else:
                id_out = self.rbr_identity(input)

            return self.nonlinearity(
                square_outputs + vertical_outputs + horizontal_outputs + id_out
            )

    def _identity_to_conv(self, identity):
        if identity is None:
            return 0, 0
        if not isinstance(identity, nn.BatchNorm2d):
            raise ValueError()
        if not hasattr(self, "id_tensor"):
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros(
                (self.in_channels, input_dim, 3, 3), dtype=np.float32
            )
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            self.id_tensor = torch.from_numpy(kernel_value).to(identity.weight.device)
        kernel = self.id_tensor
        running_mean = identity.running_mean
        running_var = identity.running_var
        gamma = identity.weight
        beta = identity.bias
        eps = identity.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        if kernel.shape[2:] == (3, 1):
            kernel = self._pad_3x1_to_3x3_tensor(kernel)
        elif kernel.shape[2:] == (1, 3):
            kernel = self._pad_1x3_to_3x3_tensor(kernel)
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel3x3_s, bias3x3_s = self._fuse_bn_tensor(self.square_conv, self.square_bn)
        kernel3x3_v, bias3x3_v = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        kernel3x3_h, bias3x3_h = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        kernelid, biasid = self._identity_to_conv(self.rbr_identity)
        kernel3x3 = kernel3x3_s + kernel3x3_v + kernel3x3_h + kernelid
        bias3x3 = bias3x3_s + bias3x3_v + bias3x3_h + biasid
        return kernel3x3, bias3x3

    def _pad_1x3_to_3x3_tensor(self, kernel1x3):
        return torch.nn.functional.pad(kernel1x3, [0, 0, 1, 1])

    def _pad_3x1_to_3x3_tensor(self, kernel3x1):
        return torch.nn.functional.pad(kernel3x1, [1, 1, 0, 0])

    def switch_to_deploy(self):
        if hasattr(self, "fused_conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(
            in_channels=self.square_conv.in_channels,
            out_channels=self.square_conv.out_channels,
            kernel_size=self.square_conv.kernel_size,
            stride=self.square_conv.stride,
            padding=self.square_conv.padding,
            dilation=self.square_conv.dilation,
            groups=self.square_conv.groups,
            bias=True,
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        for attr in [
            "square_conv",
            "square_bn",
            "ver_conv",
            "ver_bn",
            "hor_conv",
            "hor_bn",
        ]:
            self.__delattr__(attr)

        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")

    def switch_to_test(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(
            in_channels=self.square_conv.in_channels,
            out_channels=self.square_conv.out_channels,
            kernel_size=self.square_conv.kernel_size,
            stride=self.square_conv.stride,
            padding=self.square_conv.padding,
            dilation=self.square_conv.dilation,
            groups=self.square_conv.groups,
            bias=True,
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        for para in self.fused_conv.parameters():
            para.detach_()
        self.deploy = True

    def switch_to_train(self):
        if hasattr(self, "fused_conv"):
            self.__delattr__("fused_conv")
        self.deploy = False

    @staticmethod
    def build_from_config(config):
        return ACBlock(**config)


class RepConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        deploy=False,
    ):
        super(RepConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.deploy = deploy

        if len(kernel_size) != 2:
            raise ValueError()
        padding = (
            int(((kernel_size[0] - 1) * dilation) / 2),
            int(((kernel_size[1] - 1) * dilation) / 2),
        )

        self.nonlinearity = nn.ReLU(inplace=True)

        if deploy:
            self.fused_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.main_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.main_bn = nn.BatchNorm2d(num_features=out_channels)

            ver_pad = (int(((kernel_size[0] - 1) * dilation) / 2), 0)
            hor_pad = (0, int(((kernel_size[1] - 1) * dilation) / 2))

            if kernel_size[1] != 1:  # 卷积核的宽大于1 -> 有垂直卷积
                self.ver_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_size[0], 1),
                    stride=stride,
                    padding=ver_pad,
                    dilation=dilation,
                    groups=groups,
                    bias=False,
                )
                self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.ver_conv, self.ver_bn = None, None

            if kernel_size[0] != 1:  # 卷积核的高大于1 -> 有水平卷积
                self.hor_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, kernel_size[1]),
                    stride=stride,
                    padding=hor_pad,
                    dilation=dilation,
                    groups=groups,
                    bias=False,
                )
                self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.hor_conv, self.hor_bn = None, None

            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

    def forward(self, input):
        if hasattr(self, "fused_conv"):
            return self.nonlinearity(self.fused_conv(input))
        else:
            main_outputs = self.main_conv(input)
            main_outputs = self.main_bn(main_outputs)
            if self.ver_conv is not None:
                vertical_outputs = self.ver_conv(input)
                vertical_outputs = self.ver_bn(vertical_outputs)
            else:
                vertical_outputs = 0

            if self.hor_conv is not None:
                horizontal_outputs = self.hor_conv(input)
                horizontal_outputs = self.hor_bn(horizontal_outputs)
            else:
                horizontal_outputs = 0

            if self.rbr_identity is None:
                id_out = 0
            else:
                id_out = self.rbr_identity(input)

            return self.nonlinearity(
                main_outputs + vertical_outputs + horizontal_outputs + id_out
            )

    def _identity_to_conv(self, identity):
        if identity is None:
            return 0, 0
        if not isinstance(identity, nn.BatchNorm2d):
            raise ValueError()
        if not hasattr(self, "id_tensor"):
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros(
                (self.in_channels, input_dim, 1, 1), dtype=np.float32
            )
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 0, 0] = 1
            id_tensor = torch.from_numpy(kernel_value).to(identity.weight.device)
            self.id_tensor = self._pad_to_mxn_tensor(id_tensor)
        kernel = self.id_tensor
        running_mean = identity.running_mean
        running_var = identity.running_var
        gamma = identity.weight
        beta = identity.bias
        eps = identity.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        kernel = self._pad_to_mxn_tensor(kernel)
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel_mxn, bias_mxn = self._fuse_bn_tensor(self.main_conv, self.main_bn)
        if self.ver_conv is not None:
            kernel_mx1, bias_mx1 = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        else:
            kernel_mx1, bias_mx1 = 0, 0
        if self.hor_conv is not None:
            kernel_1xn, bias_1xn = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        else:
            kernel_1xn, bias_1xn = 0, 0
        kernel_id, bias_id = self._identity_to_conv(self.rbr_identity)
        kernel_mxn = kernel_mxn + kernel_mx1 + kernel_1xn + kernel_id
        bias_mxn = bias_mxn + bias_mx1 + bias_1xn + bias_id
        return kernel_mxn, bias_mxn

    def _pad_to_mxn_tensor(self, kernel):
        kernel_height, kernel_width = self.kernel_size
        height, width = kernel.shape[2:]
        pad_left_right = (kernel_width - width) // 2
        pad_top_down = (kernel_height - height) // 2
        return torch.nn.functional.pad(
            kernel, [pad_left_right, pad_left_right, pad_top_down, pad_top_down]
        )

    def switch_to_deploy(self):
        if hasattr(self, "fused_conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(
            in_channels=self.main_conv.in_channels,
            out_channels=self.main_conv.out_channels,
            kernel_size=self.main_conv.kernel_size,
            stride=self.main_conv.stride,
            padding=self.main_conv.padding,
            dilation=self.main_conv.dilation,
            groups=self.main_conv.groups,
            bias=True,
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        self.deploy = True
        for para in self.parameters():
            para.detach_()
        for attr in [
            "main_conv",
            "main_bn",
            "ver_conv",
            "ver_bn",
            "hor_conv",
            "hor_bn",
        ]:
            if hasattr(self, attr):
                self.__delattr__(attr)

        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")

    def switch_to_test(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(
            in_channels=self.main_conv.in_channels,
            out_channels=self.main_conv.out_channels,
            kernel_size=self.main_conv.kernel_size,
            stride=self.main_conv.stride,
            padding=self.main_conv.padding,
            dilation=self.main_conv.dilation,
            groups=self.main_conv.groups,
            bias=True,
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        for para in self.fused_conv.parameters():
            para.detach_()
        self.deploy = True

    def switch_to_train(self):
        if hasattr(self, "fused_conv"):
            self.__delattr__("fused_conv")
        self.deploy = False

    @staticmethod
    def is_zero_layer():
        return False

    @property
    def module_str(self):
        return "Rep_%dx%d" % (self.kernel_size[0], self.kernel_size[1])

    @property
    def config(self):
        return {
            "name": RepConvLayer.__name__,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "dilation": self.dilation,
            "groups": self.groups,
        }

    @staticmethod
    def build_from_config(config):
        return RepConvLayer(**config)
