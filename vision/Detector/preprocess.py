import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor

#BILINEAR = Image.Resampling.BILINEAR
BILINEAR = Image.BILINEAR


class Preprocess():

    def __init__(self, size=[414, 414], platform="torch"):
        if platform == "torch":
            self.preprocess = transforms.Compose([Resize(size=(size[0], size[1])), ToTensor()])
        else:
            raise ValueError(f'Not implemented for platform: {platform}')

    def __call__(self, image):
        return self.preprocess(image)


class Resize:

    def __init__(self, size=(414, 414), method=BILINEAR, dtype=np.uint8):
        self.size = size
        self.method = method
        self.dtype = dtype

    def __call__(self, input_):
        img = Image.fromarray(input_)
        return np.array(img.resize(self.size, resample=self.method)).astype(self.dtype)

