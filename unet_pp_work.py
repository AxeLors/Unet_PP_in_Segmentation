import torch
import torch.nn as nn


class StandardUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layer(inputs)

class Final_layer(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=n_classes, kernel_size=(1, 1, 1), padding='same'),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return self.layer(inputs)

class NestNet(nn.Module):
    def __init__(self, in_channels, n_classes, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        filters = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.final = Final_layer(in_channels = filters[0], n_classes=n_classes)

        # j == 0
        self.x_00 = StandardUnit(in_channels=in_channels, out_channels=filters[0])

        self.x_01 = StandardUnit(in_channels=filters[0] * 3, out_channels=filters[0])
        self.x_02 = StandardUnit(in_channels=filters[0] * 4, out_channels=filters[0])
        self.x_03 = StandardUnit(in_channels=filters[0] * 5, out_channels=filters[0])
        self.x_04 = StandardUnit(in_channels=filters[0] * 6, out_channels=filters[0])

        # j == 1
        self.x_10 = StandardUnit(in_channels=filters[0], out_channels=filters[1])

        self.x_11 = StandardUnit(in_channels=filters[1] * 3, out_channels=filters[1])
        self.x_12 = StandardUnit(in_channels=filters[1] * 4, out_channels=filters[1])
        self.x_13 = StandardUnit(in_channels=filters[1] * 5, out_channels=filters[1])

        # j == 2
        self.x_20 = StandardUnit(in_channels=filters[1], out_channels=filters[2])

        self.x_21 = StandardUnit(in_channels=filters[2] * 3, out_channels=filters[2])
        self.x_22 = StandardUnit(in_channels=filters[2] * 4, out_channels=filters[2])

        # j == 3
        self.x_30 = StandardUnit(in_channels=filters[2], out_channels=filters[3])

        self.x_31 = StandardUnit(in_channels=filters[3] * 3, out_channels=filters[3])

        # j == 4
        self.x_40 = StandardUnit(in_channels=filters[3], out_channels=filters[4])

    def forward(self, inputs, L=4):
        if not (1 <= L <= 4):
            raise ValueError("the model pruning factor `L` should be 1 <= L <= 4")

        x_00_output = self.x_00(inputs)
        x_10_output = self.x_10(self.pool(x_00_output))
        x_10_up_sample = self.up(x_10_output)
        x_01_output = self.x_01(torch.cat([x_00_output, x_10_up_sample], 1))
        nestnet_output_1 = self.final(x_01_output)

        if L == 1:
            return nestnet_output_1

        x_20_output = self.x_20(self.pool(x_10_output))
        x_20_up_sample = self.up(x_20_output)
        x_11_output = self.x_11(torch.cat([x_10_output, x_20_up_sample], 1))
        x_11_up_sample = self.up(x_11_output)
        x_02_output = self.x_02(torch.cat([x_00_output, x_01_output, x_11_up_sample], 1))
        nestnet_output_2 = self.final(x_02_output)

        if L == 2:
            if self.deep_supervision:
                # return the average of output layers
                return (nestnet_output_1 + nestnet_output_2) / 2
            else:
                return nestnet_output_2

        x_30_output = self.x_30(self.pool(x_20_output))
        x_30_up_sample = self.up(x_30_output)
        x_21_output = self.x_21(torch.cat([x_20_output, x_30_up_sample], 1))
        x_21_up_sample = self.up(x_21_output)
        x_12_output = self.x_12(torch.cat([x_10_output, x_11_output, x_21_up_sample], 1))
        x_12_up_sample = self.up(x_12_output)
        x_03_output = self.x_03(torch.cat([x_00_output, x_01_output, x_02_output, x_12_up_sample], 1))
        nestnet_output_3 = self.final(x_03_output)

        if L == 3:
            # return the average of output layers
            if self.deep_supervision:
                return (nestnet_output_1 + nestnet_output_2 + nestnet_output_3) / 3
            else:
                return nestnet_output_3

        x_40_output = self.x_40(self.pool(x_30_output))
        x_40_up_sample = self.up(x_40_output)
        x_31_output = self.x_31(torch.cat([x_30_output, x_40_up_sample], 1))
        x_31_up_sample = self.up(x_31_output)
        x_22_output = self.x_22(torch.cat([x_20_output, x_21_output, x_31_up_sample], 1))
        x_22_up_sample = self.up(x_22_output)
        x_13_output = self.x_13(torch.cat([x_10_output, x_11_output, x_12_output, x_22_up_sample], 1))
        x_13_up_sample = self.up(x_13_output)
        x_04_output = self.x_04(torch.cat([x_00_output, x_01_output, x_02_output, x_03_output, x_13_up_sample], 1))
        nestnet_output_4 = self.final(x_04_output)

        if L == 4:
            if self.deep_supervision:
                # return the average of output layers
                return (nestnet_output_1 + nestnet_output_2 + nestnet_output_3 + nestnet_output_4) / 4
            else:
                return nestnet_output_4

'''if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inputs = (-1) * torch.rand((1, 1, 32, 32, 32))
    inputs = inputs.to(device)

    ones = torch.ones_like(inputs)
    zeros = torch.zeros_like(inputs)

    model = NestNet(in_channels=1, n_classes=1, deep_supervision=False)
    model.to(device)

    from datetime import datetime

    st = datetime.now()
    output = model.forward(inputs, L=4)
    output = torch.sigmoid(output)
    res = torch.where(output > 0.5, ones, zeros)
    print(f"{(datetime.now() - st).total_seconds(): .4f}s")
'''