"""This is dataset.py from pytorch-examples.

Refer to

https://github.com/pytorch/examples/blob/master/super_resolution/dataset.py.
"""
import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
from torchvision.datasets import CelebA
import os
import PIL

def _is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def _load_img(filepath, RGB=True):
    img = Image.open(filepath)
    if RGB:
        pass
    else:
        img = img.convert('YCbCr')
        img, _, _ = img.split()
    return img


class DatasetFromFolder(data.Dataset):
    """Generate an image-to-image dataset from images from the given folder."""

    def __init__(self, image_dir, replicate=1, input_transform=None, target_transform=None, RGB=True, noise_level=0.0):
        """Init with directory, transforms and RGB switch."""
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if _is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.replicate = replicate
        self.classes = [None]
        self.RGB = RGB
        self.noise_level = noise_level

    def __getitem__(self, index):
        """Index into dataset."""
        input = _load_img(self.image_filenames[index % len(self.image_filenames)], RGB=self.RGB)
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        if self.noise_level > 0:
            # Add noise
            input += self.noise_level * torch.randn_like(input)

        return input, target

    def __len__(self):
        """Length is amount of files found."""
        return len(self.image_filenames) * self.replicate

        

from typing import Any, Callable, List, Optional, Tuple, Union

class CelebAForGender(CelebA):
    def __init__(self, root: str, split: str = "train", target_type: Union[List[str], str] = "attr", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, split, target_type, transform, target_transform, download)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X, meta =  super().__getitem__(index) 
        gender_label = meta[20]

        return X, gender_label.item()

class CelebAForMLabel(CelebA):
    def __init__(self, root: str, split: str = "train", target_type: Union[List[str], str] = "attr", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, split, target_type, transform, target_transform, download)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X, meta =  super().__getitem__(index) 

        return X, meta.to(torch.float32)


class CelebAForSmile(CelebA):
    def __init__(self, root: str, split: str = "train", target_type: Union[List[str], str] = "attr", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, split, target_type, transform, target_transform, download)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X, meta =  super().__getitem__(index) 
        smile_label = meta[31]

        return X, smile_label.item()

        

class CelebAFaceAlignForMLabel(CelebA):
    def __init__(self, root: str, split: str = "train", target_type: Union[List[str], str] = "attr", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, split, target_type, transform, target_transform, download)

    def _check_integrity(self) -> bool:
        # for (_, md5, filename) in self.file_list:
        #     fpath = os.path.join(self.root, self.base_folder, filename)
        #     _, ext = os.path.splitext(filename)
        #     # Allow original archive to be deleted (zip and 7z)
        #     # Only need the extracted images
        #     if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
        #         return False

    # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "celeba_face_align_landmarks"))
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "celeba_face_align_landmarks", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target.to(torch.float32)