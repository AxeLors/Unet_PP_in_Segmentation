import torch.nn as nn
import torch

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, target):
        bce =  (- target * torch.log(input)).mean()
        smooth = 1e-5
        num = target.size(0)
        #input = torch.where(input > 0.5, torch.ones_like(input), torch.zeros_like(input))
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2 * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice
