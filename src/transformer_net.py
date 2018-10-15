"""
original version from:
https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/transformer_net.py
commit: 645c7c386e62d2fb1d50f4621c1a52645a13869f
"""

import torch


def get_norm_layer(layer_type: str):
    if layer_type == "instance":
        return torch.nn.InstanceNorm2d
    elif layer_type == "batch":
        return torch.nn.BatchNorm2d
    elif layer_type == "none":
        return EmptyLayer
    else:
        raise ValueError("invalid normalization layer: {}".format(layer_type))


class TransformerNet(torch.nn.Module):
    def __init__(self, norm_layer=torch.nn.InstanceNorm2d, input_channels=3, coord_conv=False):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(input_channels, 32, kernel_size=9, stride=1, coord_conv=coord_conv)
        self.in1 = norm_layer(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2, coord_conv=coord_conv)
        self.in2 = norm_layer(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2, coord_conv=coord_conv)
        self.in3 = norm_layer(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2, coord_conv=coord_conv)
        self.in4 = norm_layer(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2, coord_conv=coord_conv)
        self.in5 = norm_layer(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1, coord_conv=coord_conv)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class EmptyLayer(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, coord_conv=False):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        conv = CoordConv if coord_conv else torch.nn.Conv2d
        self.conv2d = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, norm_layer=torch.nn.InstanceNorm2d, coord_conv=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, coord_conv=coord_conv)
        self.in1 = norm_layer(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, coord_conv=coord_conv)
        self.in2 = norm_layer(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, coord_conv=False):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(mode='nearest', scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        conv = CoordConv if coord_conv else torch.nn.Conv2d
        self.conv2d = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

####################################################################################
# CoordConv implementation from https://github.com/mkocabas/CoordConv-pytorch      #
####################################################################################


class AddCoords(torch.nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = torch.nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret
