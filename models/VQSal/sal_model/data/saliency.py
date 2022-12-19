## --------------------------------------------------------------------------
## Saliency in Augmented Reality
## Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
## ACM International Conference on Multimedia (ACM MM 2022)
## --------------------------------------------------------------------------

import os
import random
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset

import random
import glob
from .base import ConcatDatasetWithIndex

# from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


def DataAugTwins(img1, img2, transform):
    H, W, C = img1.shape
    img1_out = img1.copy()
    img2_out = img2.copy()
    # random_num = random.random()
    # if random_num < 0.5:
    #     img1_out[:,:int(W/2)] = img1[:,int(W/2):]
    #     img1_out[:,int(W/2):] = img1[:,:int(W/2)]
    #     img2_out[:,:int(W/2)] = img2[:,int(W/2):]
    #     img2_out[:,int(W/2):] = img2[:,:int(W/2)]
    processed = transform(image=img1_out,mask=img2_out)
    img1_out = processed["image"]
    img2_out = processed["mask"]
    
    return img1_out, img2_out

def DataAugTriplets(img1, img2, img3, transform):
    H, W, C = img1.shape
    img1_out = img1.copy()
    img2_out = img2.copy()
    img3_out = img3.copy()
    # random_num = random.random()
    # if random_num < 0.5:
    #     img1_out[:,:int(W/2)] = img1[:,int(W/2):]
    #     img1_out[:,int(W/2):] = img1[:,:int(W/2)]
    #     img2_out[:,:int(W/2)] = img2[:,int(W/2):]
    #     img2_out[:,int(W/2):] = img2[:,:int(W/2)]
    processed = transform(image=img1_out,mask=img2_out,coord=img3_out)
    img1_out = processed["image"]
    img2_out = processed["mask"]
    img3_out = processed["coord"]
    
    return img1_out, img2_out, img3_out


class SaliencyBase(Dataset):
    def __init__(self,
                 img_root, saliency_root, data_csv=None,
                 size=None, random_crop=False, interpolation="bicubic",
                 augmentation=False, is_train=False, replace=None, coord=False
                 ):
        self.data_csv = data_csv    # training/test split file
        self.img_root = img_root
        self.saliency_root = saliency_root
        if data_csv is not None:    # only load images in split file
            with open(self.data_csv, "r") as f:
                self.image_paths = f.read().splitlines()
        else:
            self.image_paths = glob.glob(os.path.join(self.img_root, '*.png'))
            self.image_paths += glob.glob(os.path.join(self.img_root, '*.jpg'))
        self.image_names = []
        for image_path in self.image_paths:
            self.image_names.append(os.path.basename(image_path))

        self._length = len(self.image_paths)

        if replace is not None:
            self.labels = {
                "relative_file_path_": [l for l in self.image_names],
                "file_path_": [os.path.join(self.img_root, l)
                            for l in self.image_names],
                "saliency_path_": [os.path.join(self.saliency_root, l.replace(replace[0], replace[1]))
                                    for l in self.image_names]
            }
        else:
            self.labels = {
                "relative_file_path_": [l for l in self.image_names],
                "file_path_": [os.path.join(self.img_root, l)
                            for l in self.image_names],
                "saliency_path_": [os.path.join(self.saliency_root, l.replace(".png", "_salMap.png"))
                                    for l in self.image_names]
            }

        size = None if size is not None and size<=0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            # keeping the aspect ratio
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.saliency_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                    interpolation=self.interpolation) #cv2.INTER_NEAREST)
            
            self.center_crop = not random_crop
            self.random_crop = random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper

        # for augmentation
        self.augmentation = augmentation
        self.transform = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.2),
            albumentations.VerticalFlip(p=0.2),
            albumentations.RandomRotate90(p=0.2),
        ])
        self.is_train = is_train

        self.coord = coord
        # if set "coord" with "True" value, which means conditional input, a "coord" key is added here
        if self.coord:
            self.transform = albumentations.Compose([self.transform],
                                                additional_targets={"coord": "image"})
            self.preprocessor = albumentations.Compose([self.preprocessor],
                                                additional_targets={"coord": "image"})

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
        
        saliency = Image.open(example["saliency_path_"])
        saliency = np.array(saliency).astype(np.uint8)
        if self.size is not None:
            saliency = self.saliency_rescaler(image=saliency)["image"]

        if not self.coord:
            # data augmentation
            if self.augmentation:
                image,saliency = DataAugTwins(image,saliency,self.transform)

            if self.size is not None and self.random_crop:
                processed = self.preprocessor(image=image,
                                            mask=saliency
                                            )
            else:   # no crop
                processed = {"image": image,
                            "mask": saliency
                            }
        else:
            h,w,_ = image.shape
            coord = np.arange(h*w).reshape(h,w,1)/(h*w)
            # data augmentation
            if self.augmentation:
                image,saliency,coord = DataAugTriplets(image,saliency,coord,self.transform)

            if self.size is not None and self.random_crop:
                processed = self.preprocessor(image=image,
                                            mask=saliency,
                                            coord=coord
                                            )
            else:
                processed = {"image": image,
                            "mask": saliency,
                            "coord": coord
                            }
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        saliency = (processed["mask"]/255).astype(np.float32)
        # saliency = processed["mask"]
        # onehot = np.eye(self.n_labels)[saliency]
        example["saliency"] = saliency #onehot
        if self.coord:
            example["coord"] = processed["coord"]
        return example


class SaliencyTrain(SaliencyBase):
    def __init__(self, img_root, saliency_root, data_csv=None, size=None, random_crop=False, interpolation="bicubic", augmentation=True, is_train=True, replace=[".jpg",".png"]):
        super().__init__(img_root=img_root,
                         saliency_root=saliency_root,
                         data_csv=data_csv,
                         size=size, random_crop=random_crop, interpolation=interpolation, augmentation=augmentation, is_train=True, replace=[".jpg",".png"])
        self.resize = albumentations.Resize(height=int(size), width=int(size), interpolation=self.interpolation) # for some datasets, after rescaler, width/height == 1 may failed
        
    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        saliency = Image.open(example["saliency_path_"])
        saliency = np.array(saliency).astype(np.uint8)

        if len(saliency.shape) < 3:
            saliency = saliency[:, :, np.newaxis]
        if saliency.shape[2]>1:
            saliency = np.mean(saliency,2)  # rgb->gray

        if self.size is not None:
            # image = self.image_rescaler(image=image)["image"]
            # saliency = self.saliency_rescaler(image=saliency)["image"]

            image = self.resize(image=image)["image"]
            saliency = self.resize(image=saliency)["image"]
        
        # data augmentation
        if self.augmentation:
            image,saliency = DataAugTwins(image,saliency,self.transform)
        if self.size is not None and self.is_train and self.random_crop:
            processed = self.preprocessor(image=image,
                                          mask=saliency)
        else:
            processed = {"image": image,
                         "mask": saliency}
        image = (processed["image"]/127.5 - 1.0).astype(np.float32)
        saliency = (processed["mask"]/255).astype(np.float32)

        example["image"] = image
        example["saliency"] = saliency #onehot
        return example


class SaliencyValidation(SaliencyBase):
    def __init__(self, img_root, saliency_root, data_csv=None, size=None, random_crop=False, interpolation="bicubic", augmentation=False, is_train=False, replace=[".jpg",".png"]):
        super().__init__(img_root=img_root,
                         saliency_root=saliency_root,
                         data_csv=data_csv,
                         size=size, random_crop=random_crop, interpolation=interpolation, augmentation=augmentation, is_train=is_train, replace=replace)
        self.resize = albumentations.Resize(height=int(size), width=int(size), interpolation=self.interpolation) # for some datasets, after rescaler, width/height == 1 may failed

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        saliency = Image.open(example["saliency_path_"])
        saliency = np.array(saliency).astype(np.uint8)

        if len(saliency.shape) < 3:
            saliency = saliency[:, :, np.newaxis]
        if saliency.shape[2]>1:
            saliency = np.mean(saliency,2)  # rgb->gray

        if self.size is not None:
            # image = self.image_rescaler(image=image)["image"]
            # saliency = self.saliency_rescaler(image=saliency)["image"]

            image = self.resize(image=image)["image"]
            saliency = self.resize(image=saliency)["image"]
        
        image = (image/127.5 - 1.0).astype(np.float32)
        saliency = (saliency/255).astype(np.float32)

        example["image"] = image
        example["saliency"] = saliency #onehot
        return example





def rescale_allimgs(source_path, write_path, size):
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
    for filename in os.listdir(source_path):
        print(filename)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(source_path,filename))
            if not img.mode == "RGB":
                img = img.convert("RGB")
            img = np.array(img).astype(np.uint8)
            img = image_rescaler(image=img)["image"]
            img = Image.fromarray(img)
            img.save(os.path.join(write_path,filename.replace('.jpg', '.png')))


def split_train_validation(source_path, write_path, name='', ratio=6):
    filelist = os.listdir(source_path)
    print(len(filelist))
    sublist = random.sample(filelist, int(len(filelist)/ratio))
    for i, sfile in enumerate(filelist):
        print(sfile)
        if sfile in sublist:
            with open(os.path.join(write_path,name+'_validation.txt'), "a+") as f:
                f.write(sfile)
                f.write("\n")
        else:
            with open(os.path.join(write_path,name+'_train.txt'), "a+") as f:
                f.write(sfile)
                f.write("\n")
        print(i)
