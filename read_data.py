import SimpleITK as sitk
import os
from torch.utils.data import Dataset
import numpy as np

def set_ww_wl(sitk_image, ww, wl):
    window_filter = sitk.IntensityWindowingImageFilter()
    window_filter.SetOutputMaximum(255)
    window_filter.SetOutputMinimum(0)
    window_filter.SetWindowMinimum(wl-ww//2)
    window_filter.SetWindowMaximum(wl+ww//2)
    result = window_filter.Execute(sitk_image)

    cast_filter = sitk.CastImageFilter()
    cast_filter.SetOutputPixelType(sitk.sitkUInt8)
    result = cast_filter.Execute(result)

    return result

def resample(wait_img, new_size):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(wait_img)  # 需要重新采样的目标图像
    size = wait_img.GetSize()
    spacing = wait_img.GetSpacing()
    new_spacing = (spacing[0] * size[0] / new_size[0], spacing[1] * size[1] / new_size[1], spacing[2] * size[2] / new_size[2])
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkLinear)
    itk_img = resampler.Execute(wait_img)
    return itk_img

def image_roi(sitk_image, index, size):
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetIndex(index)
    roi_filter.SetSize(size)
    result = roi_filter.Execute(sitk_image)
    return result

def read_nrrd(nrrd_path):
    wait_img = sitk.ReadImage(nrrd_path)
    # 原始imag信息
    wait_info_ls = [wait_img.GetSize(), wait_img.GetOrigin(), wait_img.GetDirection(), wait_img.GetSpacing()]

    size = wait_info_ls[0]
    # cut_img = image_roi(wait_img, index=(128, 128, int((size[2])//3)), size=(256, 256, int((2*size[2])//3)))
    cut_img = image_roi(wait_img, index=(192, 192, int((size[2])//4)), size=(128, 128, int((3*size[2])//4)))

    img_final = resample(cut_img, (48, 48, 64))
    return img_final, wait_info_ls


class My_dataset2(Dataset):
    def __init__(self, src_root_path):
        super().__init__()
        self.case_list = os.listdir(src_root_path)
        self.src_root_path = src_root_path
        pass

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, index):
        one_case_path = self.case_list[index]
        image_data_path = os.path.join(self.src_root_path, one_case_path, 'img.nrrd')
        sitk_image, _ = read_nrrd(image_data_path)
        sitk_image = set_ww_wl(sitk_image, 300, 150)
        image_volume_data = sitk.GetArrayFromImage(sitk_image)
        image_volume_data = image_volume_data.astype(np.float32)
        shape = image_volume_data.shape
        x_image = image_volume_data.reshape((1, shape[0], shape[1], shape[2]))

        label_data_path = os.path.join(self.src_root_path, one_case_path, 'structures', 'BrainStem.nrrd')
        sitk_label, _ = read_nrrd(label_data_path)
        label_volume_data = sitk.GetArrayFromImage(sitk_label)
        label_volume_data = label_volume_data.astype(np.float32)
        y_label = label_volume_data.reshape((1, shape[0], shape[1], shape[2]))

        pid = os.path.basename(self.case_list[index])
        return pid, x_image, y_label

