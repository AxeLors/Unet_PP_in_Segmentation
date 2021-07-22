from unet_pp_work import NestNet
import torch
from torch.utils.data import DataLoader
from loss_fun import BinaryDiceLoss
from read_data import My_dataset2
import os
import SimpleITK as sitk
import numpy as np
from read_data import resample
import pandas as pd

# 原始imag信息
#wait_info_ls = [wait_img.GetSize(), wait_img.GetOrigin(), wait_img.GetDirection(), wait_img.GetSpacing()]
def postprocess(pred_itk, wait_info_ls):
    '''
    预处理第一步：resample
    预处理第二步：roi的逆运算, 即填充
    '''
    ori_size = eval(wait_info_ls[0])
    resample_size = (128, 128, int((3*ori_size[2]))//4)
    resample_itk = resample(pred_itk, resample_size)
    resample_ay = sitk.GetArrayFromImage(resample_itk)
    binary_ay = np.where(resample_ay > 0.5, np.ones_like(resample_ay), np.zeros_like(resample_ay))

    wait_given_ay = np.zeros((ori_size[2], ori_size[1], ori_size[0]))
    for i in range(int((3 * ori_size[2]) // 4)):
        for j in range(128):
            for k in range(128):
                wait_given_ay[i + int((ori_size[2]) // 4), j + 192, k + 192] = binary_ay[i, j, k]
    result_itk = sitk.GetImageFromArray(wait_given_ay)
    result_itk.SetSpacing(eval(wait_info_ls[3]))
    result_itk.SetOrigin(eval(wait_info_ls[1]))
    result_itk.SetDirection(eval(wait_info_ls[2]))
    return result_itk

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_path = 'data'
data_test = My_dataset2(os.path.join(root_path, 'PDDCA-1.4_part3'))
data_loader_test = DataLoader(data_test, batch_size=1, shuffle=False)

Model = NestNet(in_channels = 1, n_classes = 1, deep_supervision=False).to(device)
para_path = 'paras/2021_07_21_17_55_55.784399_RESUnet.pt'
Model.load_state_dict(torch.load(para_path))
criterion = BinaryDiceLoss()

info_path = 'Part3_info.csv'
df = pd.read_csv(info_path)
info_ay = np.asarray(df)


loss_ls = []

for i, (pid, inputs, labels) in enumerate(data_loader_test):
    #print(i)
    x_image = torch.as_tensor(inputs).to(device)
    y_label = torch.as_tensor(labels).to(device)
    pred = Model.forward(x_image, 4)
    loss = criterion(pred, y_label)
    loss_ls.append(loss.item())
    pred = torch.reshape(pred, [pred.shape[2], pred.shape[3], pred.shape[4]])
    pred = pred.data.cpu().numpy()
    pred_itk = sitk.GetImageFromArray(pred)
    #对每一个pred做后处理，让这个张量最终变成一个nrrd文件
    wait_info_ls = list(info_ay[i])
    result_itk = postprocess(pred_itk, wait_info_ls)
    sitk.WriteImage(result_itk, 'pred_nrrd/RESUnet2/' + str(pid[0]) + '.nrrd')

loss_mse = sum(loss_ls) / len(loss_ls)