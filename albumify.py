import time
import torch
from torchvision import transforms
import albumentations
import albumentations.pytorch
import numpy as np


class Albumify:
    """
    Class to take one interpolated image and create an album from it of images that have been transformed by rotations and Flips.
    For now, we keep it very small but we might introduce more transformations as we go along.

    Code developed by IC.
    """

    def __init__(self, savedir=None, transforms=[]):

        self.savedir = savedir
        self.transforms = transforms

    def __call__(self, name, img):

        path = (
            self.savedir / (name + "_album.npy") if self.savedir is not None else None
        )

        if path is not None and path.exists():
            return np.load(path, allow_pickle="True").item()

        album = self.__get_album(img)

        if path is not None:
            np.save(path, album)

        return album

    def __get_album(self, img):
        """Function to take in an image and do the magic."""
        # transforms = {
        #     "HorizontalFlip": albumentations.HorizontalFlip(p=1),
        #     "VerticalFlip": albumentations.VerticalFlip(p=1),
        #     "Transpose": albumentations.Transpose(p=1),
        #     "ShiftScaleRotate": albumentations.ShiftScaleRotate(p=1),
        # }

        album = {}
        album["interp"] = img

        for strans in self.transforms:
            print('Albumify using', strans, 'transform')
            trans = getattr(albumentations, strans)

            augmented = trans(p=1)(image=img)
            image = augmented["image"]
            album[strans] = image

        # if self.do_transform:
        #     for key, transformer in transforms.items():
        #         augmented = transformer(image=img)  #
        #         image = augmented["image"]
        #         album[key] = image

        return album
