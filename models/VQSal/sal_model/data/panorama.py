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

class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex

class PanoramaBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None, random_crop=False, interpolation="bicubic",
                 augmentation=False, is_train=False, suff='.jpg', coord=False
                 ):
        self.data_root = data_root
        self.image_paths = glob.glob(os.path.join(self.data_root, '*'+suff)) #[glob.glob(os.path.join(l, '*'+suff)) for l in self.data_root]
        
        self._length = len(self.image_paths)
        self.labels = {
            "file_path_": [l for l in self.image_paths]
        }

        self.coord = coord
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
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.center_crop = not random_crop
            if self.center_crop:    # may delete it since we don't use it in panorama datasets
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper
            
            # if set "coord" with "True" value, which means conditional input, a "coord" key is added here
            if self.coord:
                self.preprocessor = albumentations.Compose([self.preprocessor],
                                                    additional_targets={"coord": "image"})

        # for augmentation
        self.augmentation = augmentation
        self.transform = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.2),
            albumentations.VerticalFlip(p=0.2),
            albumentations.RandomRotate90(p=0.2),
        ])
        self.random_crop = random_crop
        self.is_train = is_train

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
        
        if self.size is not None and self.random_crop: #self.is_train: # crop during training
            # processed = self.preprocessor(image=image)
            if not self.coord:  # only return image
                processed = self.preprocessor(image=image)
                example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
            else:   # return "image" and "coord"
                h,w,_ = image.shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                processed = self.preprocessor(image=image, coord=coord)
                example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
                example["coord"] = processed["coord"]
        else:   # keep size during validation
            # processed = {"image": image}
            if not self.coord:  # only return image
                processed = {"image": image}
                example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
            else:   # return "image" and "coord"
                h,w,_ = image.shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w) # [0,h*w] reshape to [h,w]
                processed = {"image": image, "coord": coord}
                example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
                example["coord"] = processed["coord"]
        # example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        return example

class PanoramaTrain(Dataset):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", augmentation=True, is_train=True, coord=False):
        d1 = PanoramaBase(data_root="data/panoramas/pano65000_train/SUN360_train",
                         size=size, random_crop=random_crop, interpolation=interpolation, augmentation=augmentation, is_train=is_train, coord=coord)
        d2 = PanoramaBase(data_root="data/panoramas/pano65000_train/StreetView_train",
                         size=size, random_crop=random_crop, interpolation=interpolation, augmentation=augmentation, is_train=is_train, coord=coord)
        self.data = ConcatDatasetWithIndex([d1, d2])
        # super().__init__(data_root=["data/panoramas/pano65000_train/SUN360_train","data/panoramas/pano65000_train/StreetView_train"],
        #                  size=size, random_crop=random_crop, interpolation=interpolation, augmentation=augmentation, is_train=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        return ex


class PanoramaValidation(Dataset):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", coord=False):
        # super().__init__(data_root=["data/panoramas/pano65000_test/SUN360_tset","data/panoramas/pano65000_test/StreetView_test"],
        #                  size=size, random_crop=random_crop, interpolation=interpolation)
        d1 = PanoramaBase(data_root="data/panoramas/pano65000_test/SUN360_tset",
                         size=size, random_crop=random_crop, interpolation=interpolation, coord=coord)
        d2 = PanoramaBase(data_root="data/panoramas/pano65000_test/StreetView_test",
                         size=size, random_crop=random_crop, interpolation=interpolation, coord=coord)
        self.data = ConcatDatasetWithIndex([d1, d2])
        # super().__init__(data_root=["data/panoramas/pano65000_train/SUN360_train","data/panoramas/pano65000_train/StreetView_train"],
        #                  size=size, random_crop=random_crop, interpolation=interpolation, augmentation=augmentation, is_train=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        
        return ex


class PanoramaValidation_sub(Dataset):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", coord=False):
        # super().__init__(data_root=["data/panoramas/pano65000_test/SUN360_tset","data/panoramas/pano65000_test/StreetView_test"],
        #                  size=size, random_crop=random_crop, interpolation=interpolation)
        d1 = PanoramaBase(data_root="data/panoramas/pano65000_test/SUN360_subtset",
                         size=size, random_crop=random_crop, interpolation=interpolation, coord=coord)
        d2 = PanoramaBase(data_root="data/panoramas/pano65000_test/StreetView_subtest",
                         size=size, random_crop=random_crop, interpolation=interpolation, coord=coord)
        self.data = ConcatDatasetWithIndex([d1, d2])
        # super().__init__(data_root=["data/panoramas/pano65000_train/SUN360_train","data/panoramas/pano65000_train/StreetView_train"],
        #                  size=size, random_crop=random_crop, interpolation=interpolation, augmentation=augmentation, is_train=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        print(ex["file_path_"])
        return ex










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


class PanoramaSaliencyBase(Dataset):
    def __init__(self,
                 data_csv, data_root, saliency_root,
                 size=None, random_crop=False, interpolation="bicubic",
                 augmentation=False, is_train=False, replace=None, coord=False
                 ):
        self.data_csv = data_csv
        self.data_root = data_root
        self.saliency_root = saliency_root
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)

        if replace is not None:
            self.labels = {
                "relative_file_path_": [l for l in self.image_paths],
                "file_path_": [os.path.join(self.data_root, l)
                            for l in self.image_paths],
                "saliency_path_": [os.path.join(self.saliency_root, l.replace(replace[0], replace[1]))
                                    for l in self.image_paths]
            }
        else:
            self.labels = {
                "relative_file_path_": [l for l in self.image_paths],
                "file_path_": [os.path.join(self.data_root, l)
                            for l in self.image_paths],
                "saliency_path_": [os.path.join(self.saliency_root, l.replace(".png", "_salMap.png"))
                                    for l in self.image_paths]
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
        # assert segmentation.mode == "L", segmentation.mode
        saliency = np.array(saliency).astype(np.uint8)
        # if self.shift_segmentation:
        #     # used to support segmentations containing unlabeled==255 label
        #     segmentation = segmentation+1
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
            else:
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


class PanoramaSaliencyASDTrain(PanoramaSaliencyBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", augmentation=True, is_train=True):
        super().__init__(data_csv="data/panoramas/panoramas_train.txt",
                         data_root="data/panoramas/pano300_512",
                         saliency_root="data/panoramas/pano300_ASD",
                         size=size, random_crop=random_crop, interpolation=interpolation, augmentation=augmentation, is_train=True)


class PanoramaSaliencyASDValidation(PanoramaSaliencyBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv="data/panoramas/panoramas_validation.txt",
                         data_root="data/panoramas/pano300_512",
                         saliency_root="data/panoramas/pano300_ASD",
                         size=size, random_crop=random_crop, interpolation=interpolation)


class PanoramaSaliencyTDTrain(PanoramaSaliencyBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", augmentation=True, is_train=True, coord=False):
        super().__init__(data_csv="data/panoramas/panoramas_train.txt",
                         data_root="data/panoramas/pano300_512",
                         saliency_root="data/panoramas/pano300_TD",
                         size=size, random_crop=random_crop, interpolation=interpolation, augmentation=augmentation, is_train=True, coord=coord)


class PanoramaSaliencyTDValidation(PanoramaSaliencyBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", coord=False):
        super().__init__(data_csv="data/panoramas/panoramas_validation.txt",
                         data_root="data/panoramas/pano300_512",
                         saliency_root="data/panoramas/pano300_TD",
                         size=size, random_crop=random_crop, interpolation=interpolation, coord=coord)


class PanoramaAOITrain(PanoramaSaliencyBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", augmentation=True, is_train=True):
        super().__init__(data_csv="data/panoramas/AOI_train.txt",
                         data_root="data/panoramas/AOI_Dataset/img_600",
                         saliency_root="data/panoramas/AOI_Dataset/salmapn1",
                         size=size, random_crop=random_crop, interpolation=interpolation, augmentation=augmentation, is_train=True, replace=[".jpg","_sal.jpg"])
        self.resize = albumentations.Resize(height=int(size), width=int(size*2), interpolation=self.interpolation) # for some datasets, after rescaler, width/height == 1 may failed
    
    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        saliency = Image.open(example["saliency_path_"])
        saliency = np.array(saliency).astype(np.uint8)

        # *** add following code for this dataset ***
        if saliency.shape[2]>1:
            saliency = np.mean(saliency,2)  # rgb->gray
        # *** add following code for this dataset ***

        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
            saliency = self.saliency_rescaler(image=saliency)["image"]

        # *** add following code for this dataset ***
        image = self.resize(image=image)["image"]
        saliency = self.resize(image=saliency)["image"]
        # *******************************************
        
        # data augmentation
        if self.augmentation:
            image,saliency = DataAugTwins(image,saliency,self.transform)
        if self.size is not None and self.is_train:
            processed = self.preprocessor(image=image,
                                          mask=saliency
                                          )
        else:
            processed = {"image": image,
                         "mask": saliency
                         }
        image = (processed["image"]/127.5 - 1.0).astype(np.float32)
        saliency = (processed["mask"]/255).astype(np.float32)

        example["image"] = image
        example["saliency"] = saliency #onehot
        return example


class PanoramaAOIValidation(PanoramaSaliencyBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv="data/panoramas/AOI_validation.txt",
                         data_root="data/panoramas/AOI_Dataset/img_600",
                         saliency_root="data/panoramas/AOI_Dataset/salmapn1",
                         size=size, random_crop=random_crop, interpolation=interpolation, replace=[".jpg","_sal.jpg"])
        # self.base_dataset = PanoramaSaliencyBase(data_csv="data/panoramas/AOI_validation.txt",
        #                  data_root="data/panoramas/AOI_Dataset/img_600",
        #                  saliency_root="data/panoramas/AOI_Dataset/salmapn1",
        #                  size=size, random_crop=random_crop, interpolation=interpolation, replace=[".jpg","_sal.jpg"])
        self.resize = albumentations.Resize(height=int(size), width=int(size*2), interpolation=self.interpolation) # for some datasets, after rescaler, width/height == 1 may failed
    

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        saliency = Image.open(example["saliency_path_"])
        saliency = np.array(saliency).astype(np.uint8)

        # *** add following code for this dataset ***
        if saliency.shape[2]>1:
            saliency = np.mean(saliency,2)  # rgb->gray
        # *** add following code for this dataset ***

        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
            saliency = self.saliency_rescaler(image=saliency)["image"]

        # *** add following code for this dataset ***
        image = self.resize(image=image)["image"]
        saliency = self.resize(image=saliency)["image"]
        # *******************************************
        
        # data augmentation
        if self.augmentation:
            image,saliency = DataAugTwins(image,saliency,self.transform)
        if self.size is not None and self.is_train:
            processed = self.preprocessor(image=image,
                                          mask=saliency
                                          )
        else:
            processed = {"image": image,
                         "mask": saliency
                         }
        image = (processed["image"]/127.5 - 1.0).astype(np.float32)
        saliency = (processed["mask"]/255).astype(np.float32)

        example["image"] = image
        example["saliency"] = saliency #onehot
        return example





class PanoramaSalRecTrain(Dataset):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", augmentation=True, is_train=True, coord=False):
        d1 = PanoramaSaliencyTDTrain(size=size, random_crop=random_crop, interpolation=interpolation)
        d2 = PanoramaAOITrain(size=size, random_crop=random_crop, interpolation=interpolation)
        self.data = ConcatDatasetWithIndex([d1, d2])
        # super().__init__(data_root=["data/panoramas/pano65000_train/SUN360_train","data/panoramas/pano65000_train/StreetView_train"],
        #                  size=size, random_crop=random_crop, interpolation=interpolation, augmentation=augmentation, is_train=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        return ex


class PanoramaSalRecValidation(Dataset):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic", coord=False):
        # super().__init__(data_root=["data/panoramas/pano65000_test/SUN360_tset","data/panoramas/pano65000_test/StreetView_test"],
        #                  size=size, random_crop=random_crop, interpolation=interpolation)
        d1 = PanoramaSaliencyTDValidation(size=size, random_crop=random_crop, interpolation=interpolation)
        d2 = PanoramaAOIValidation(size=size, random_crop=random_crop, interpolation=interpolation)
        self.data = ConcatDatasetWithIndex([d1, d2])
        # super().__init__(data_root=["data/panoramas/pano65000_train/SUN360_train","data/panoramas/pano65000_train/StreetView_train"],
        #                  size=size, random_crop=random_crop, interpolation=interpolation, augmentation=augmentation, is_train=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        
        return ex







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


# if __name__ == '__main__':

    # *** use following lines to generate train/test split ***
    # rescale_allimgs('../../data/panoramas/pano300','../../data/panoramas/pano300_512',512)
    # split_train_validation('../../data/panoramas/pano300_512','../../data/panoramas','panoramas',6)
    # split_train_validation('../../data/panoramas/AOI_Dataset/img_600','../../data/panoramas','AOI',6)
    # ********************************************************

    # *** use following lines to test if some images are broken ***
    # data_root = "../../data/panoramas/pano65000_train/SUN360_train"
    # data_root = "../../data/panoramas/pano65000_train/StreetView_train"
    # data_root = "../../data/panoramas/pano65000_test/SUN360_test"
    # data_root = "../../data/panoramas/pano65000_test/StreetView_test"
    # suff = '.jpg'
    # image_paths = glob.glob(os.path.join(data_root, '*'+suff)) #[glob.glob(os.path.join(l, '*'+suff)) for l in self.data_root]
    # i = 0
    # for path in image_paths:
    #     i = i+1
    #     print(i)
    #     print(path)
    #     image = Image.open(path)
    #     if not image.mode == "RGB":
    #         image = image.convert("RGB")
    #     image = np.array(image).astype(np.uint8)
    # *************************************************************
    