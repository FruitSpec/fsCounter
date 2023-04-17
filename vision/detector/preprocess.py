import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor
import cv2
#BILINEAR = Image.Resampling.BILINEAR
BILINEAR = Image.BILINEAR


class Preprocess():

    def __init__(self, device, size=[414, 414], platform="torch"):
        if platform == "torch":
            self.transform = transforms.Compose([Resize_(size=(size[0], size[1])), ToTensor()])
            self.device = device
        else:
            raise ValueError(f'Not implemented for platform: {platform}')

    def __call__(self, image):

        preprc_frame = self.transform(image)
        #preprc_frame = torch.unsqueeze(preprc_frame, dim=0)
        preprc_frame = preprc_frame.to(self.device)

        return preprc_frame


class Resize_:

    def __init__(self, size=(414, 414), swap=(2, 0, 1), method=BILINEAR, dtype=np.uint8):
        self.size = size
        self.method = method
        self.dtype = dtype
        self.swap = swap

    def __call__(self, input_):

        if len(input_.shape) == 3:
            padded_img = np.ones((self.size[0], self.size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.size, dtype=np.uint8) * 114

        r = min(self.size[0] / input_.shape[0], self.size[1] / input_.shape[1])
        resized_img = cv2.resize(
            input_,
            (int(input_.shape[1] * r), int(input_.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(input_.shape[0] * r), : int(input_.shape[1] * r)] = resized_img

        #padded_img = padded_img.transpose(self.swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img


        #img = Image.fromarray(input_)
        #return np.array(img.resize(self.size, resample=self.method)).astype(self.dtype)

