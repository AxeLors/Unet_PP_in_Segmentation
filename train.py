from unet_pp_work import NestNet
import torch
from torch.utils.data import DataLoader
from loss_fun import BinaryDiceLoss
from read_data import My_dataset2
import os
import datetime
from VGG_pp_work import UNet
import time

'''
本项目中X_train, Y_train等的维度为[sample_num, 1, 80, 256, 256]
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = 'data'
data_train = My_dataset2(os.path.join(root_path, 'PDDCA-1.4_part1&2'))
data_loader_train = DataLoader(data_train, batch_size=1, shuffle=True)

Model = NestNet(in_channels = 1, n_classes = 1, deep_supervision=False).to(device)

criterion = BinaryDiceLoss()
optimizer = torch.optim.Adam(Model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1, last_epoch=-1)

epochs = 40

for epoch in range(epochs):
    for i, (pid, inputs, labels) in enumerate(data_loader_train):
        x_image = torch.as_tensor(inputs).to(device)
        y_label = torch.as_tensor(labels).to(device)
        optimizer.zero_grad()
        outputs = Model.forward(x_image, 4)
        loss = criterion(outputs, y_label)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        timestr = time.strftime('%H:%M:%S', time.localtime())
        print('%8s %10s: [%04d/%04d]===>[loss: %.6f, accuracy: %.6f]' % (
            timestr, pid[0], epoch + 1, i + 1, loss.item(), 1 - loss.item()))

def save_paras():
    path = 'paras/' + str(datetime.datetime.today()).replace(' ', '_').replace(':', '_').replace('-', '_') + '_RESUnet.pt'
    torch.save(Model.state_dict(), path)
    pass

save_paras()