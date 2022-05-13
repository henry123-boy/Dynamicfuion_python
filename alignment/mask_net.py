import torch
import torch.nn as nn
from alignment.nn_utilities import make_conv_2d, ResBlock2d, Identity


class MaskNet(nn.Module):
    def __init__(self, use_batch_normalization: bool = False):

        super().__init__()

        fn_0 = 16
        self.input_fn = fn_0 + 6 * 2
        fn_1 = 16

        self.upconv1 = torch.nn.ConvTranspose2d(in_channels=565, out_channels=2 * fn_0, kernel_size=4, stride=2,
                                                padding=1)
        self.upconv2 = torch.nn.ConvTranspose2d(in_channels=2 * fn_0, out_channels=fn_0, kernel_size=4, stride=2,
                                                padding=1)

        if use_batch_normalization:
            custom_batch_norm = torch.nn.BatchNorm2d
        else:
            custom_batch_norm = Identity

        self.model = nn.Sequential(
            make_conv_2d(self.input_fn, fn_1, n_blocks=1, normalization=custom_batch_norm),
            ResBlock2d(fn_1, normalization=custom_batch_norm),
            ResBlock2d(fn_1, normalization=custom_batch_norm),
            ResBlock2d(fn_1, normalization=custom_batch_norm),
            nn.Conv2d(fn_1, 1, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, features, x):
        # Reduce number of channels and upscale to highest resolution
        features = self.upconv1(features)
        features = self.upconv2(features)

        x = torch.cat([features, x], 1)
        assert x.shape[1] == self.input_fn

        return self.model(x)
