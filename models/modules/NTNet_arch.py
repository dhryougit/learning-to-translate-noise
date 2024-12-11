# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# from basicsr.models.archs.arch_util import LayerNorm2d
# from basicsr.models.archs.local_arch import Local_Base
# from basicsr.utils.flops_util import count_model_param_flops, print_model_param_nums

# from basicsr.models.archs.quant_ops  import quantize, quantize_grad, QConv2d, QLinear, RangeBN

# from .module_util import SinusoidalPosEmb, LayerNorm, exists


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


def exists(x):
    return x is not None


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        net_bias = True
        self.norm1 = LayerNorm(c)
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=net_bias,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=net_bias,
        )
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=net_bias,
            ),
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=net_bias,
        )
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.norm2 = LayerNorm(c)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=net_bias,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=net_bias,
        )
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class NTNet(nn.Module):

    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1],
        noise_injection_level=100,
    ):
        super().__init__()

        net_bias = True
        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=net_bias,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=net_bias,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.mask = {}

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2, bias=net_bias))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)
        self.noise_level = noise_injection_level

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        # for gray
        if C == 1:
            inp = inp.repeat(1, 3, 1, 1)

        x = self.intro(inp)

        encs = []
        ############################################
        for encoder, down in zip(self.encoders, self.downs):
            x = x + torch.randn_like(x) * self.noise_level / 255
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = x + torch.randn_like(x) * self.noise_level / 255
        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = x + torch.randn_like(x) * self.noise_level / 255
            x = decoder(x)

        x = x + torch.randn_like(x) * self.noise_level / 255

        x = self.ending(x)

        x = x + inp


        if C == 1:
            x = x.mean(dim=1).unsqueeze(dim=1)
        return torch.clamp(x[:, :, :H, :W], 0, 1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == "__main__":
    img_channel = 3
    net = NTNet(
        img_channel=img_channel,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1],
    )

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info
    from torchsummary import summary as summary_

    summary_(net.cuda(), (3, 256, 256), batch_size=1)

    macs, params = get_model_complexity_info(
        net, inp_shape, verbose=False, print_per_layer_stat=True
    )

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
