## --------------------------------------------------------------------------
## Saliency in Augmented Reality
## Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
## ACM International Conference on Multimedia (ACM MM 2022)
## --------------------------------------------------------------------------

from os import write
import sys
sys.path.append(".")

# also disable grad to save memory
import torch
torch.set_grad_enabled(False)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
# from sal_model.models.vqgan import VQSalModel
from sal_model.models.vqsal_ar import VQSalModel

import PIL
from PIL import Image

import numpy as np

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None):
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def load_vqgan_sal(config, ckpt_path=None):
    model = VQSalModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def custom_to_pil_sal(x):
    x = x.detach().cpu()
    print(x.shape)
    x = torch.clamp(x, 0., 1.)
    x = x[0].numpy()  # use x[0], since it's grayscale
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    # if not x.mode == "RGB":
    #   x = x.convert("RGB")
    return x


import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from dall_e          import map_pixels, unmap_pixels, load_model
def preprocess(img, target_image_size=256):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    print(2 * [target_image_size])
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)

import albumentations
import cv2
from PIL import Image
image_rescaler = albumentations.SmallestMaxSize(max_size=512, interpolation=cv2.INTER_AREA)
# image_resizer = albumentations.Resize(height=int(256), width=int(256), interpolation=cv2.INTER_AREA)
image_resizer = albumentations.Resize(height=int(512), width=int(512), interpolation=cv2.INTER_AREA)
def preprocess2(img):
    image = Image.open(img)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    # image = image_rescaler(image=image)["image"]
    image = image_resizer(image=image)["image"]
    image = (image/127.5 - 1.0).astype(np.float32)
    image = torch.unsqueeze(T.ToTensor()(image), 0)
    return map_pixels(image)#image

def reconstruct_with_vqgan(x, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    z, _, [_, _, indices] = model.encode(x)
    print(f"VQGAN: latent shape: {z.shape[2:]}")
    xrec = model.decode(z)
    return xrec

def reconstruct_with_vqgan_sal(input_Superimposed, input_AR, input_BG, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    quant, diff, _ = model.encode(input_Superimposed)
    quant_AR, diff_AR, _ = model.encode(input_AR)
    quant_BG, diff_BG, _ = model.encode(input_BG)
    dec = model.decoder(model.post_quant_conv(quant))
    dec = model.sal_conv(dec)
    dec_AR = model.decoder_AR(model.post_quant_conv(quant_AR))
    dec_AR = model.sal_conv_AR(dec_AR)
    dec_BG = model.decoder_BG(model.post_quant_conv(quant_BG))
    dec_BG = model.sal_conv_AR(dec_BG)
    dec = model.sal_conv_final(torch.cat((dec_AR, dec, dec_BG), 1))
    
    return dec

def cal_sal(input_Superimposed, input_AR, input_BG, model_sal):
    # x = preprocess(PIL.Image.open(path), target_image_size=size)
    input_Superimposed = preprocess2(input_Superimposed)
    input_AR = preprocess2(input_AR)
    input_BG = preprocess2(input_BG)
    input_Superimposed = input_Superimposed.to(DEVICE)
    input_AR = input_AR.to(DEVICE)
    input_BG = input_BG.to(DEVICE)
    x_sal = reconstruct_with_vqgan_sal(input_Superimposed, input_AR, input_BG, model_sal)
    return x_sal






import os
import glob
import argparse

def get_args():
    parser = argparse.ArgumentParser('VQSal prediction', add_help=False)
    parser.add_argument('--config_path', type=str, help='config file path of model')
    parser.add_argument('--model_path', type=str, help='checkpoint path of model')
    parser.add_argument('--input_path', type=str, help='input images path')
    parser.add_argument('--output_path', type=str, help='output images path')
    parser.add_argument('--data_csv', type=str, default=None, help='output images path')

    return parser.parse_args()

# -------------------------------------------- main test code below ---------------------------------------------
if __name__ == '__main__':
    args = get_args()

    # config1024 = load_config("logs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
    config_sal = load_config(args.config_path, display=False)

    # model1024 = load_vqgan(config1024, ckpt_path="logs/2021-05-29T22-11-23_panorama_vqgan/checkpoints/last.ckpt").to(DEVICE) 
    model_sal = load_vqgan_sal(config_sal, ckpt_path=args.model_path).to(DEVICE)

    # x,x_sal = reconstruction_pipeline(path='data/test/2_P2_5000x2500.png', size=512)

    # print(x.shape)
    # x = custom_to_pil(x[0])
    # x_sal = custom_to_pil_sal(x_sal[0])
    # x = x.save("test.jpg")
    # x_sal = x_sal.save("test_sal.jpg")

    input_path = args.input_path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)  
    data_csv = args.data_csv
    if data_csv is not None:
        with open(data_csv, "r") as f:
            file_lines = f.read().splitlines()
        # img_paths = [os.path.join(input_path, l) for l in image_names] # glob.glob(r'F:\Resources\CV\taming-transformers-master_\data\panoramas\pano300_512\*.png')
    else:
        img_paths = glob.glob(os.path.join(input_path, '*.png'))
        img_paths += glob.glob(os.path.join(input_path, '*.jpg'))

    image_names = []
    image_paths = []
    image_paths_ar = []
    image_paths_bg = []
    cnt = 0
    for image_name in file_lines:
        cnt = cnt+1
        if cnt == 1:
            continue
        name = image_name.split(',')
        image_names.append(name[2])    # superimposed images
        image_paths.append(os.path.join(input_path, 'Superimposed', name[2]))   # superimposed images
        image_paths_ar.append(os.path.join(input_path, 'AR', name[1]))    # AR images
        image_paths_bg.append(os.path.join(input_path, 'BG', name[0]))    # BG images

    cnt=0
    
    for (image_name, image_path, image_path_ar, image_path_bg) in zip(image_names, image_paths, image_paths_ar, image_paths_bg):
        cnt=cnt+1
        print(cnt)
        print(image_path)
        name = image_name
        x_sal = cal_sal(input_Superimposed=image_path, input_AR=image_path_ar, input_BG=image_path_bg, model_sal=model_sal)
        x_sal = custom_to_pil_sal(x_sal[0])
        x_sal = x_sal.save(os.path.join(output_path,name))
        # x,x_sal = reconstruction_pipeline(path=path, size=256)
        # x = custom_to_pil(x[0])
        # x_sal = custom_to_pil_sal(x_sal[0])
        # x = x.save(os.path.join(write_rec_path,name))
        # x_sal = x_sal.save(os.path.join(output_path,name))
        # raw = Image.open(path)
        # raw = raw.save(os.path.join(write_raw_path,name))