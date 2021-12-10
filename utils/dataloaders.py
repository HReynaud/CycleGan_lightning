from logging import warning
import os
from torchio.constants import DATA
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import scipy
import json
from typing import Type
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

import torchio as tio
import elasticdeform as ed

SEGMENTATION_DICT = {
    "Background": 0,
    "Liver": 1,
    "Bladder": 2,
    "Lungs": 3,
    "Kidneys": 4,
    "Bone": 0,  # bones are ignored
    "Brain": 0,  # brain is ignored
}


def format_ct(ct):
    ct = (ct - ct.min()) / (ct.max() - ct.min())
    ct = ct.astype(np.float16)
    return ct


def format_xcat_seg(seg):
    seg = re_segment(seg)
    seg = seg.astype(np.uint8)
    return seg


def format_ct_seg(seg):
    seg[seg == 5] = 0  # ignore bone
    seg = seg.astype(np.uint8)
    return seg


def re_segment(seg):
    new_seg = np.zeros(seg.shape)
    new_seg[seg == 1228] = SEGMENTATION_DICT["Liver"]
    new_seg[seg == 1394] = SEGMENTATION_DICT["Bladder"]

    new_seg[seg == 1224] = SEGMENTATION_DICT["Lungs"]
    new_seg[seg == 1222] = SEGMENTATION_DICT["Lungs"]
    new_seg[np.logical_and(seg >= 1746, seg <= 2143)] = SEGMENTATION_DICT["Lungs"]

    new_seg[seg == 1269] = SEGMENTATION_DICT["Kidneys"]
    new_seg[seg == 1270] = SEGMENTATION_DICT["Kidneys"]

    return new_seg


def Identity(x):
    return x


def save_image(image, filename):
    im_range = image.max() - image.min()
    if im_range > 0:
        image = (image - image.min()) / im_range * 255  # normalize
    Image.fromarray(image.astype(np.uint8)).save(filename)


class StyleTransferLoaderV1(LightningDataModule):
    def __init__(
        self,
        XCAT_volumes_path,
        CT_volumes_path,
        lazy=True,
        hashtable_path="./",
        slice_dim=512,
        batch_size=1,
    ):
        super().__init__()
        self.train_ds = FourOrganDataset(
            XCAT_volumes_path,
            CT_volumes_path,
            lazy=True,
            split="train",
            hashtable_path=hashtable_path,
            deterministic_xcat_idx=False,
            slice_dim=512,
            spatial_transform={"None": None},
            # spatial_transform={
            #     "xcat": {
            #         "sigma": 25,
            #         "points": 3,
            #     },
            #     "ct": {
            #         "sigma": 25,
            #         "points": 3,
            #     },
            # },
        )
        self.val_ds = FourOrganDataset(
            XCAT_volumes_path,
            CT_volumes_path,
            slice_dim=slice_dim,
            split="val",
            lazy=True,
            hashtable_path=hashtable_path,
        )
        self.preload = not lazy
        self.bs = batch_size
        self.num_workers = min(os.cpu_count(), self.bs)

        if self.preload:
            warning("Preloading data will delay experiment start.")
            self.train_ds.load_all()
            self.val_ds.load_all()

    def setup(self, stage=None):
        # if self.preload:
        #     self.train_ds.load_all()
        #     self.val_ds.load_all()
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )


class StyleTransferLoaderV2(LightningDataModule):
    def __init__(
        self,
        dataset_path="/vol/biomedic3/hjr119/StyleTransfer/DATA/FOUR_ORGANS",
        slice_dim=512,
        batch_size=1,
        spatial_transform={"None": None},
        load_seg=False,
    ):
        super().__init__()
        self.train_ds = FourOrganSlices(
            dataset_path,
            split="train",
            deterministic_xcat_idx=False,
            slice_dim=slice_dim,
            spatial_transform=spatial_transform,
            filter_arm_out=False,
            load_seg=load_seg,
        )
        self.val_ds = FourOrganSlices(
            dataset_path,
            split="val",
            deterministic_xcat_idx=False,
            slice_dim=slice_dim,
            spatial_transform=spatial_transform,
            filter_arm_out=False,
            load_seg=load_seg,
        )
        self.bs = batch_size
        self.num_workers = min(os.cpu_count(), self.bs)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )


class StyleTransferLoaderHZ(LightningDataModule):
    def __init__(
        self,
        dataset_path="/vol/biomedic3/hjr119/StyleTransfer/DATA/datasets/horse2zebra",
        batch_size=1,
        load_seg=False,
    ):
        super().__init__()
        self.train_ds = Horse2Zebras(dataset_path, split="train")
        self.val_ds = Horse2Zebras(dataset_path, split="val")
        self.bs = batch_size
        self.num_workers = min(os.cpu_count(), self.bs)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )


class StyleTransferLoaderV3(LightningDataModule):
    def __init__(
        self,
        dataset_path="/vol/biomedic3/hjr119/StyleTransfer/DATA/FOUR_ORGANS_256",
        batch_size=1,
        img_size=256,
    ):
        super().__init__()
        self.train_ds = FourOrganImages(dataset_path, split="train", slice_dim=img_size)
        self.val_ds = FourOrganImages(dataset_path, split="val", slice_dim=img_size)
        self.bs = batch_size
        self.num_workers = min(os.cpu_count(), self.bs)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )


class FourOrganDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        XCAT_volumes_path,
        CT_volumes_path,
        lazy=True,
        split="train",
        hashtable_path="./",
        deterministic_xcat_idx=False,
        slice_dim=512,
        spatial_transform={"None": None},
    ):
        self.xc_path = XCAT_volumes_path
        self.ct_path = CT_volumes_path
        self.xc_deterministic_idx = deterministic_xcat_idx
        self.slice_dim = slice_dim
        self.spatial_transform = spatial_transform

        assert split in ["train", "val", "test"]
        # self.len = 47718 if split=='train' else 15785

        self.idx_to_slice = json.load(
            open(os.path.join(hashtable_path, "hashtable_" + split + ".json"))
        )

        # Load XCAT data on ram:
        xcat_list = os.listdir(self.xc_path)
        xcat_list = [xcat[:-9] for xcat in xcat_list if xcat.endswith("CT.nii.gz")]
        xcat_list.sort()

        if split == "train":
            xcat_list = xcat_list[: int(len(xcat_list) * 0.8)]
        else:
            xcat_list = xcat_list[int(len(xcat_list) * 0.8) :]

        self.xcat_subjects = []
        for xcat_name in tqdm(xcat_list):
            ct_name = xcat_name + "CT.nii.gz"
            seg_name = xcat_name + "SEG.nii.gz"

            self.xcat_subjects.append(
                LazyVolumeLoader(
                    os.path.join(self.xc_path, ct_name),
                    os.path.join(self.xc_path, seg_name),
                    format_xcat_seg,
                    xcat_name[:-1],
                )
            )
            if not lazy:
                self.xcat_subjects[-1].load()

        # load CT data on ram:
        ct_list = os.listdir(self.ct_path)
        ct_list.sort()

        if split == "train":
            ct_list = ct_list[: int(len(ct_list) * 0.8)]
        else:
            ct_list = ct_list[int(len(ct_list) * 0.8) :]

        self.ct_subjects = []
        for volume_name in tqdm(ct_list):
            ct_l_path = os.path.join(self.ct_path, volume_name)
            seg_l_path = ct_l_path.split("/")
            seg_l_path[-2] = "labels"
            seg_l_path[-1] = seg_l_path[-1].replace("volume", "labels")
            seg_l_path = os.path.join("/", *seg_l_path)

            self.ct_subjects.append(
                LazyVolumeLoader(ct_l_path, seg_l_path, format_ct_seg, volume_name[:-7])
            )
            if not lazy:
                self.ct_subjects[-1].load()

    def __getitem__(self, idx):

        # XCAT - use random idx
        if self.xc_deterministic_idx:
            xc_idx = idx % (len(self.xcat_subjects) * self.slice_dim)
            xc_v_idx = xc_idx // self.slice_dim
            xc_s_idx = xc_idx % self.slice_dim
        else:
            xc_v_idx = torch.randint(len(self.xcat_subjects), (1,)).item()
            xc_s_idx = torch.randint(self.slice_dim, (1,)).item()

        xcat_volume_slice = self.xcat_subjects[xc_v_idx].ct[xc_s_idx]
        xcat_seg_slice = self.xcat_subjects[xc_v_idx].seg[xc_s_idx]
        if "xcat" in self.spatial_transform.keys():
            # transform = self.get_random_spatial_transform()
            # xcat_volume_slice, = transform(xcat_volume_slice[None,:,:,:])[0,:,:,:]
            # xcat_seg_slice = transform(xcat_seg_slice[None,:,:,:])[0,:,:,:]
            xcat_volume_slice, xcat_seg_slice = ed.deform_random_grid(
                [
                    xcat_volume_slice.astype(np.float32),
                    xcat_seg_slice.astype(np.float32),
                ],
                sigma=self.spatial_transform["xcat"]["sigma"],
                points=self.spatial_transform["xcat"]["points"],
            )
            xcat_volume_slice = xcat_volume_slice.astype(np.float16)
            xcat_seg_slice = xcat_seg_slice.view(dtype=np.float16)

        # CT - use hashed idx
        ct_v_idx, ct_s_idx = self.idx_to_slice[str(idx)]
        ct_volume_slice = self.ct_subjects[ct_v_idx].ct[ct_s_idx]
        ct_seg_slice = self.ct_subjects[ct_v_idx].seg[ct_s_idx]
        if "ct" in self.spatial_transform.keys():
            # transform = self.get_random_spatial_transform()
            # ct_volume_slice = transform(ct_volume_slice[None,:,:,:])[0,:,:,:]
            # ct_seg_slice = transform(ct_seg_slice[None,:,:,:])[0,:,:,:]
            ct_volume_slice, ct_seg_slice = ed.deform_random_grid(
                [
                    ct_volume_slice.astype(np.float32),
                    ct_seg_slice.astype(np.float32),
                ],
                sigma=self.spatial_transform["ct"]["sigma"],
                points=self.spatial_transform["ct"]["points"],
            )
            ct_volume_slice = ct_volume_slice.astype(np.float16)
            ct_seg_slice = ct_seg_slice.view(dtype=np.float16)

        return xcat_volume_slice, xcat_seg_slice, ct_volume_slice, ct_seg_slice

    def __len__(self):
        return len(self.idx_to_slice)

    def get_random_spatial_transform(self):
        cp = 7
        md = np.rint(self.slice_dim / cp / 5)
        transform = tio.transforms.Compose(
            [
                tio.transforms.RandomElasticDeformation(
                    num_control_points=cp, max_displacement=md, locked_borders=2
                ),
            ]
        )

        return transform

    def load_all(self):
        for subject in tqdm(self.xcat_subjects, desc="Loading XCAT"):
            subject.load()
        for subject in tqdm(self.ct_subjects, desc="Loading CT"):
            subject.load()


class FourOrganSlices(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        split="train",
        deterministic_xcat_idx=False,
        slice_dim=512,
        spatial_transform={"None": None},
        filter_arm_out=False,
        load_seg=False,
    ):
        self.split_percentage = 0.8
        self.root = dataset_path
        self.split = split
        self.deterministic_xcat_idx = deterministic_xcat_idx
        self.slice_dim = slice_dim
        self.spatial_transform = spatial_transform
        self.filter_arm_out = filter_arm_out
        self.load_seg = load_seg

        self.image_path = "IMAGES"
        self.seg_path = "SEGS"

        self.xcat_path = os.path.join(self.root, "XCAT", self.image_path)
        self.ct_path = os.path.join(self.root, "CT", self.image_path)

        self.xcat_list = os.listdir(self.xcat_path)
        self.xcat_list.sort()
        if self.split == "train":
            self.xcat_list = self.xcat_list[
                : int(self.split_percentage * len(self.xcat_list))
            ]
        elif self.split == "val":
            self.xcat_list = self.xcat_list[
                int(self.split_percentage * len(self.xcat_list)) :
            ]

        self.ct_list = os.listdir(self.ct_path)
        self.ct_list.sort()
        if self.split == "train":
            self.ct_list = self.ct_list[
                : int(self.split_percentage * len(self.ct_list))
            ]
        elif self.split == "val":
            self.ct_list = self.ct_list[
                int(self.split_percentage * len(self.ct_list)) :
            ]

    def _load_ct_im(self, idx, dim=512.0):
        slice_path = os.path.join(self.root, "CT", self.image_path, self.ct_list[idx])
        arr = np.load(slice_path).astype(np.float)
        arr = scipy.ndimage.zoom(arr, dim / 512.0)
        return arr

    def _load_xcat_im(self, idx, dim=512.0):
        slice_path = os.path.join(
            self.root, "XCAT", self.image_path, self.xcat_list[idx]
        )
        arr = np.load(slice_path).astype(np.float)
        if self.filter_arm_out:
            raise NotImplementedError(
                "Filterings the arms from XCAT is not implemented yet."
            )
        arr = scipy.ndimage.zoom(arr, dim / 512.0)
        return arr

    def _load_ct_SEG(self, idx, dim=512.0):
        slice_path = os.path.join(self.root, "CT", self.seg_path, self.ct_list[idx])
        arr = np.load(slice_path).astype(np.float)
        arr[arr == 5] = 0  # ignore bone
        arr = scipy.ndimage.zoom(arr, dim / 512.0)
        return arr

    def _load_xcat_SEG(self, idx, dim=512.0):
        slice_path = os.path.join(self.root, "XCAT", self.seg_path, self.xcat_list[idx])
        arr = np.load(slice_path)
        arr = re_segment(arr).astype(np.float)
        arr = scipy.ndimage.zoom(arr, dim / 512.0)
        return arr

    def __getitem__(self, idx):
        ct_im = self._load_ct_im(idx, self.slice_dim)
        ct_seg = (
            self._load_ct_SEG(idx, self.slice_dim)
            if self.load_seg
            else np.zeros_like(ct_im)
        )

        xcat_idx = (
            idx % len(self.xcat_list)
            if self.deterministic_xcat_idx
            else np.random.randint(0, len(self.xcat_list))
        )
        xcat_im = self._load_xcat_im(xcat_idx, self.slice_dim)
        xcat_seg = (
            self._load_xcat_SEG(xcat_idx, self.slice_dim)
            if self.load_seg
            else np.zeros_like(xcat_im)
        )

        if "xcat" in self.spatial_transform.keys():
            xcat_im, xcat_seg = ed.deform_random_grid(
                [xcat_im, xcat_seg],
                sigma=self.spatial_transform["xcat"]["sigma"],
                points=self.spatial_transform["xcat"]["points"],
            )

        if "ct" in self.spatial_transform.keys():
            ct_im, ct_seg = ed.deform_random_grid(
                [ct_im, ct_seg],
                sigma=self.spatial_transform["ct"]["sigma"],
                points=self.spatial_transform["ct"]["points"],
            )

        return (
            xcat_im[None, ...],
            ct_im[None, ...],
            xcat_seg[None, ...],
            ct_seg[None, ...],
        )

    def __len__(self):
        return len(self.ct_list)


class Horse2Zebras(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        split="train",
    ):
        self.split = "train" if split == "train" else "test"
        self.dataset_path = dataset_path
        self.horses = os.listdir(os.path.join(dataset_path, self.split + "A"))
        self.zebras = os.listdir(os.path.join(dataset_path, self.split + "B"))
        self.horses.sort()
        self.zebras.sort()
        for i in range(len(self.horses)):
            self.horses[i] = os.path.join(
                dataset_path, self.split + "A", self.horses[i]
            )
        for i in range(len(self.zebras)):
            self.zebras[i] = os.path.join(
                dataset_path, self.split + "B", self.zebras[i]
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize([286, 286], Image.BICUBIC),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                # transforms.Lambda(lambda img: img / 255),
            ]
        )

    def __getitem__(self, idx):
        index_Z = np.random.randint(0, len(self.zebras))
        index_H = idx
        image_H = self.transform(Image.open(self.horses[index_H]).convert("RGB"))
        image_Z = self.transform(Image.open(self.zebras[index_Z]).convert("RGB"))
        return (
            image_H,
            image_Z,
            image_H,
            image_Z,
        )

    def __len__(self):
        return len(self.horses)


class FourOrganImages(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        split="train",
        slice_dim=256,
    ):
        xcat_distribution = {
            "mean": 60.65102218091488,
            "std": 84.46573544008183,
        }
        ct_distribution = {
            "mean": 51.89822253520426,
            "std": 45.26092384768775,
        }
        self.split = split
        self.dataset_path = dataset_path
        self.xcat = os.listdir(os.path.join(dataset_path, "XCAT", "IMAGES"))
        self.ct = os.listdir(os.path.join(dataset_path, "CT", "IMAGES"))
        self.xcat.sort()
        self.ct.sort()

        self.xcat = (
            self.xcat[: int(len(self.xcat) * 0.8)]
            if split == "train"
            else self.xcat[int(len(self.xcat) * 0.8) :]
        )
        self.ct = (
            self.ct[: int(len(self.ct) * 0.8)]
            if split == "train"
            else self.ct[int(len(self.ct) * 0.8) :]
        )

        for i in range(len(self.xcat)):
            self.xcat[i] = os.path.join(dataset_path, "XCAT", "IMAGES", self.xcat[i])
        for i in range(len(self.ct)):
            self.ct[i] = os.path.join(dataset_path, "CT", "IMAGES", self.ct[i])

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    [np.rint(slice_dim * 1.5), np.rint(slice_dim * 1.5)], Image.BICUBIC
                ),
                transforms.RandomCrop(slice_dim),
                transforms.PILToTensor(),
            ]
        )
        self.transform_xcat = transforms.Compose(
            [
                self.transform,
                transforms.Lambda(
                    lambda img: (img - xcat_distribution["mean"])
                    / xcat_distribution["std"]
                ),
            ]
        )
        self.transform_ct = transforms.Compose(
            [
                self.transform,
                transforms.Lambda(
                    lambda img: (img - ct_distribution["mean"]) / ct_distribution["std"]
                ),
            ]
        )

    def __getitem__(self, idx):
        index_X = np.random.randint(0, len(self.xcat))
        index_C = idx
        image_X = self.transform_xcat(Image.open(self.xcat[index_X]).convert("L"))
        image_C = self.transform_ct(Image.open(self.ct[index_C]).convert("L"))
        return (
            image_X,
            image_C,
            image_X,
            image_C,
        )

    def __len__(self):
        return len(self.ct)


class LazyVolumeLoader:
    def __init__(self, path_ct, path_seg, seg_formart_func, name="None"):
        self._path_ct = path_ct
        self._path_seg = path_seg
        self._format = seg_formart_func
        self.name = name

        self._is_transformed = {"ct": True, "seg": True}

        self._ct = None
        self._seg = None

    @property
    def ct(self):
        if self._ct is None:
            self._ct = format_ct(
                sitk.GetArrayFromImage((sitk.ReadImage(self._path_ct)))
            )
        return self._ct

    @property
    def seg(self):
        if self._seg is None:
            self._seg = self._format(
                sitk.GetArrayFromImage((sitk.ReadImage(self._path_seg)))
            )
        return self._seg

    def load(self):
        self.ct
        self.seg
        return self

    def is_loaded(self):
        return self._ct is not None and self._seg is not None

    # UNUSED
    def no_transform(self):
        if self._is_transformed["ct"]:
            self._ct = format_ct(
                sitk.GetArrayFromImage((sitk.ReadImage(self._path_ct)))
            )
            self._is_transformed["ct"] = False

        if self._is_transformed["seg"]:
            self._seg = self._format(
                sitk.GetArrayFromImage((sitk.ReadImage(self._path_seg)))
            )
            self._is_transformed["seg"] = False

        return self

    # UNUSED
    def apply_transform(self, transform):
        self._ct = transform(self.ct)
        self._seg = transform(self.seg)
        self._is_transformed["ct"] = False
        self._is_transformed["seg"] = False
        return self


def get_datamodule(args):
    if args.dataloader.lower() == "v1":
        return StyleTransferLoaderV1(
            XCAT_volumes_path=args.xcat_path,
            CT_volumes_path=args.ct_path,
            lazy=not args.preload,
            hashtable_path=args.hashtable_path,
            slice_dim=args.imdim,
            batch_size=args.batch_size,
        )
    elif args.dataloader.lower() == "v2":
        return StyleTransferLoaderV2(
            dataset_path=args.dataset_path,
            slice_dim=args.imdim,
            batch_size=args.batch_size,
            spatial_transform={"None": None},
            load_seg=args.use_seg,
        )
    elif args.dataloader.lower() == "hz":
        return StyleTransferLoaderHZ(
            dataset_path=args.dataset_path, batch_size=args.batch_size
        )
    elif args.dataloader.lower() == "v3":
        return StyleTransferLoaderV3(
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            img_size=args.imdim,
        )
    else:
        raise NotImplementedError(f"{args.dataloader} is not implemented")


if __name__ == "__main__":
    # ds = FourOrganDataset(
    #     XCAT_volumes_path="/vol/biomedic3/hjr119/XCAT/generation_MICCAI22",
    #     CT_volumes_path="/vol/biomedic3/hjr119/DATA/CTORG/volumes",
    #     lazy=True,
    #     split="train",
    #     hashtable_path="/vol/biomedic3/hjr119/StyleTransfer/utils/hashtables",
    #     deterministic_xcat_idx=False,
    #     slice_dim=512,
    #     # spatial_transform={
    #     #     "xcat": {"sigma": 7, "points": 7},
    #     #     "ct": {"sigma": 7, "points": 7},
    #     # },
    # )
    # ds.load_all()
    # print("stop")

    # dsl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=2)
    # for c, batch in tqdm(enumerate(dsl)):
    #     print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape)
    #     save_image(batch[0].numpy()[0], "./tmp/xcat_" + str(c) + ".jpg")
    #     save_image(batch[1].numpy()[0], "./tmp/xcat_seg_" + str(c) + ".jpg")
    #     save_image(batch[2].numpy()[0], "./tmp/ct_" + str(c) + ".jpg")
    #     save_image(batch[3].numpy()[0], "./tmp/ct_seg_" + str(c) + ".jpg")
    #     if c > 10:
    #         break

    # dm = StyleTransferLoaderV1(
    #     XCAT_volumes_path="/vol/biomedic3/hjr119/XCAT/generation_MICCAI22",
    #     CT_volumes_path="/vol/biomedic3/hjr119/DATA/CTORG/volumes",
    #     lazy=True,
    #     hashtable_path="/vol/biomedic3/hjr119/StyleTransfer/utils/hashtables",
    #     slice_dim=512,
    #     batch_size=2,
    # )

    dst = FourOrganSlices(
        dataset_path="/vol/biomedic3/hjr119/StyleTransfer/DATA/FOUR_ORGANS",
        split="train",
        deterministic_xcat_idx=False,
        slice_dim=128,
        spatial_transform={"None": None},
        filter_arm_out=False,
        load_seg=False,
    )
    dsv = FourOrganSlices(
        dataset_path="/vol/biomedic3/hjr119/StyleTransfer/DATA/FOUR_ORGANS",
        split="val",
        deterministic_xcat_idx=False,
        slice_dim=128,
        spatial_transform={"None": None},
        filter_arm_out=False,
        load_seg=False,
    )
    print(len(dst) + len(dsv))
