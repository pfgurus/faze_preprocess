import numpy as np
import torch
from torch import nn
from torch.hub import load_state_dict_from_url

from common import components as cc
from common import utils as cu
import cv2

from . import model

# Classes
CLASSES = [
    'background',
    'skin',
    'l_brow',
    'r_brow',
    'l_eye',
    'r_eye',
    'eye_g',  # Eyeglasses
    'l_ear',
    'r_ear',
    'ear_r',
    'nose',
    'mouth',
    'u_lip',
    'l_lip',
    'neck',
    'neck_l',  # Necklace
    'cloth',
    'hair',
    'hat'
]


class FaceParser(nn.Module):
    """ Compute a face parsing mask from 512x512 image. """
    def __init__(self):
        super().__init__()
        self._model = model.BiSeNet(19)
        url = 'https://github.com/pfgurus/Public-large-file-storage/releases/download/face_parser/79999_iter.pth'
        model_dict = load_state_dict_from_url(url=url)
        self._model.load_state_dict(model_dict)
        self._model = self._model.to(cu.set_device())

        s = (1, 3, 1, 1)
        self.register_buffer('_mean', torch.tensor(cc.TorchModels.MEAN_2).reshape(s), persistent=False)
        self.register_buffer('_std', torch.tensor(cc.TorchModels.STD_2).reshape(s), persistent=False)

        self.eval()
        self.requires_grad_(False)

    def forward(self, x):
        """ Compute a face parsing mask.
        :param x: [B, 3, S, S] RGB image in range [-1, 1], will be rescaled to 512x512 resolution.
        :return: [B, 19, 64, 64] tensor with class logits.
        """
        x = (x - self._mean) / self._std
        x = cu.interpolate(x, size=512)
        return self._model(x)

    def infer(self, img: np.ndarray):
        """ Compute a face parsing mask.
        :param img: [H,W,3] RGB image 0-255.
        :return: [H, W] Numpy array with class indices.
        """
        img_512 = cv2.resize(img, (512, 512))
        img_tensor = torch.tensor(img_512.transpose((2, 0, 1)).astype(np.float32) / 255).to(cu.set_device())
        res = self.forward(img_tensor)
        res = cu.interpolate(res, size=512)
        res = torch.argmax(res, dim=1).squeeze(0).detach().cpu().numpy()

        res_img = cv2.resize(res, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        return res_img

    def get_viz_image(self, img:np.ndarray):
        res = self.infer(img)
        res = res/np.max(res)*255
        res = cv2.applyColorMap((res).astype(np.uint8), cv2.COLORMAP_JET)
        return res
