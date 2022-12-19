## Baseline Saliency Models

Please download the baseline saliency models from this link: [baseline_saliency_models](https://github.com/DuanHuiyu/baseline_saliency_models)

This repo is the sub-repo of [ARSaliency](https://github.com/DuanHuiyu/ARSaliency).
If this repo can help you with your research, please consider citing out paper.
```
@inproceedings{duan2022saliency,
  title={Saliency in Augmented Reality},
  author={Duan, Huiyu and Shen, Wei and Min, Xiongkuo and Tu, Danyang and Li, Jing and Zhai, Guangtao},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  year={2022}
}
```

### Download Necessary Pretrain weights (for DNN only)
Download necessary pretrain weights for DNN from: [[百度网盘](https://pan.baidu.com/s/1dkH4NdMG82Tql9ll2oRwkw?pwd=yctf)] [[TeraBox](https://terabox.com/s/1n8-ZUATnDVy8RX22y5BzGQ)].

### Calculate Traditional Saliency Maps
- Demo:
```
calTraSalDemo.m
```
- Run all calculation on SARD (may need to change the path according to the annotation in this file):
```
calTraSal.m
```

### Train/Test Deep Saliency Models
Please follow the instructions in "deep_saliency".

#### 1. Installation: Pytorch (for Salicon, Sal-CFS-GAN, and VQSal)

- create environment
```
conda create -n saliency python=3.8 pip
conda activate saliency
conda install visdom dominate -c conda-forge
```
- for CUDA 11.3 users
```
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
- for CUDA 10.2 users
``` 
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=10.2 -c pytorch
```
- install requirements
```
cd deep_saliency
pip install -r requirements.txt
```

#### 2. Installation: Theano (for mlnet, sam-vgg/resnet, and salgan)

may not work for 30 series graphic cards

- create environment & install requirements
```
conda create -n saliency_theano python=3.6 pip
conda activate saliency_theano
conda install theano==1.0.2 pygpu
conda install cudnn cudatoolkit=10.2
pip install keras==1.2.0

# for mlnet
conda install h5py==2.10.0
pip install opencv-python

# for salgan
cd deep_saliency/salgan-master
pip install Lasagne-master.zip  # version: 0.2.dev1
pip install tqdm==4.5.0
```

- since we may not have root, thus we copy all include files to virtual env
```
# copy all cuda files to current anaconda environment
cp -r /usr/local/cuda/include/* /usr/local/anaconda3/envs/saliency_theano/include/
```

- change keras backend in ".keras/keras.json"
```
vim .keras/keras.json
```
change the codes the following lines
```
{
  "image_dim_ordering": "th",
  "floatx": "float32",
  "epsilon": 1e-07,
  "backend": "theano",
  "image_data_format": "channels_last"
}
```

- modify "~/.theanorc" file
```
vim ~/.theanorc
```
add the following lines (please note whether the path is right)
```
[global]
openmp=False 
device = gpu   
floatX = float32  
allow_input_downcast=True  
[lib]
cnmem = 0.8 
[blas]
ldflags= -lopenblas
[nvcc]
fastmath = True  
[cuda]
root=/usr/local/cuda/bin
[dnn]
include_path=/usr/local/anaconda3/envs/saliency_theano/include/
library_path=/usr/local/anaconda3/envs/saliency_theano/lib/
```

#### 3. Train/test on Salicon dataset

**3.1 MLNet** 
- train
```
# need to change path name in "./deep_saliency/mlnet-master/config_salicon.py"
# parameter: phase, model_save_dir
cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_salicon.py train 'salicon'
```
- test
```
# parameter: phase, imgs_test_path, model_save_dir, output_folder
cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_salicon.py test '/data/duan/sal_all/datasets/Salicon/val/images/' 'salicon' 'results/salicon/'
```

**3.2 Sal-CFS-GAN** 
- train
    
Note: Set the training/testing path in "deep_saliency/Sal-CFS-GAN-master/GazeGAN_LocalGlobal_Pytorch/options/base_options.py" ('--dataroot') and "deep_saliency/Sal-CFS-GAN-master/GazeGAN_LocalGlobal_Pytorch/data/aligned_dataset.py" (dir_A, dir_B and dir_C represent the subpaths of source image, dense saliency map, and discrete fixation map))
```
cd deep_saliency/Sal-CFS-GAN-master/
python3 train.py \
--name label2city_512p --no_instance --label_nc 0 --no_ganFeat_loss --netG global_unet --resize_or_crop none \
--dataroot "/data/duan/sal_all/datasets/Salicon/train" --batchSize 10
```
- test
```
cd deep_saliency/Sal-CFS-GAN-master/
python3 test.py --name label2city_512p --netG global_unet --ngf 64 --resize_or_crop none --label_nc 0 --no_instance \
--dataroot "/data/duan/sal_all/datasets/Salicon/val" \
--results_dir './results/salicon' \
--how_many 5000
```

**3.3 SALGAN** 
- train the model (flag, train_img_path, train_sal_path, val_img_path, val_sal_path, model_name)
```
cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python train.py 'bce' \
'/data/duan/sal_all/datasets/Salicon/train/images/' \
'/data/duan/sal_all/datasets/Salicon/train/maps/' \
'/data/duan/sal_all/datasets/Salicon/val/images/' \
'/data/duan/sal_all/datasets/Salicon/val/maps/' \
'_salicon_bce'

cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python train.py 'salgan' \
'/data/duan/sal_all/datasets/Salicon/train/images/' \
'/data/duan/sal_all/datasets/Salicon/train/maps/' \
'/data/duan/sal_all/datasets/Salicon/val/images/' \
'/data/duan/sal_all/datasets/Salicon/val/maps/' \
'_salicon_salgan'
```

- test the model (img_path, model_name, output_path)
```
cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python test.py \
'/data/duan/sal_all/datasets/Salicon/val/images/' \
'_salicon_bce' \
'results/salicon_bce'

cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python test.py \
'/data/duan/sal_all/datasets/Salicon/val/images/' \
'_salicon_salgan' \
'results/salicon_salgan'
```

**3.4 Salicon** 
- train salicon
```
cd deep_saliency/Salicon_pytorch-master/src/
python3 train.py \
--train_dataset_dir '/data/duan/sal_all/datasets/Salicon/train' \
--test_dataset_dir '/data/duan/sal_all/datasets/Salicon/val' \
--train_img_dir 'images' \
--train_label_dir 'maps' \
--batch_size 2 \
--epochs 50 \
--model_name ''
```

- test salicon
```
cd deep_saliency/Salicon_pytorch-master/src/
python3 test.py \
--test_dataset_dir '/data/duan/sal_all/datasets/Salicon/val' \
--test_img_dir 'images' \
--test_label_dir 'maps' \
--model_name '' \ 
--output_dir 'results'
```

**3.5 SAM-VGG & SAM-ResNet** 
if cannot download model automatically, download it manually and put it in '~/.keras/models/'
- train (phase, version, fixpts_type, model_dir)
```
cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_salicon.py train 0 'mat' 'sam-vgg_salicon'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_salicon.py train 1 'mat' 'sam-resnet_salicon'
```

- test (phase, version, imgs_test_path, model_dir, output_folder)
```
cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_salicon.py test 0 '/data/duan/sal_all/datasets/Salicon/val/images/' 'sam-vgg_salicon' 'results/salicon_vgg/'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_salicon.py test 1 '/data/duan/sal_all/datasets/Salicon/val/images/' 'sam-resnet_salicon' 'results/salicon_resnet/'
```


#### 4. Train/test on SARD dataset

**4.1 MLNet** 
- train superimposed (phase, model_save_dir, train_csv_file, test_csv_file)
```
cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 'ar0' 'train0.csv' 'test0.csv'

cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 'ar1' 'train1.csv' 'test1.csv'

cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 'ar2' 'train2.csv' 'test2.csv'

cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 'ar3' 'train3.csv' 'test3.csv'

cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 'ar4' 'train4.csv' 'test4.csv'
```
- test superimposed (phase, imgs_test_path, model_save_dir, output_folder)
```
cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'ar0' 'results/superimposed/' 'test0.csv'

cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'ar1' 'results/superimposed/' 'test1.csv'

cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'ar2' 'results/superimposed/' 'test2.csv'

cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'ar3' 'results/superimposed/' 'test3.csv'

cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'ar4' 'results/superimposed/' 'test4.csv'
```
- test AR (phase, imgs_test_path, model_save_dir, output_folder)
```
cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test '/data/duan/sal_all/datasets/SARD/small_size/AR/' 'salicon' 'results/AR/'
```
- test BG (phase, imgs_test_path, model_save_dir, output_folder)
```
cd deep_saliency/mlnet-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test '/data/duan/sal_all/datasets/SARD/small_size/BG/' 'salicon' 'results/BG/'
```

**4.2 Sal-CFS-GAN** 
- train Superimposed
```
cd deep_saliency/Sal-CFS-GAN-master/
python3 train.py \
--name ar0 --no_instance --label_nc 0 --no_ganFeat_loss --netG global_unet --resize_or_crop none \
--dataroot '/data/duan/sal_all/datasets/SARD' --batchSize 5 --data_csv 'train0.csv'

cd deep_saliency/Sal-CFS-GAN-master/
python3 train.py \
--name ar1 --no_instance --label_nc 0 --no_ganFeat_loss --netG global_unet --resize_or_crop none \
--dataroot '/data/duan/sal_all/datasets/SARD' --batchSize 5 --data_csv 'train1.csv'

cd deep_saliency/Sal-CFS-GAN-master/
python3 train.py \
--name ar2 --no_instance --label_nc 0 --no_ganFeat_loss --netG global_unet --resize_or_crop none \
--dataroot '/data/duan/sal_all/datasets/SARD' --batchSize 5 --data_csv 'train2.csv'

cd deep_saliency/Sal-CFS-GAN-master/
python3 train.py \
--name ar3 --no_instance --label_nc 0 --no_ganFeat_loss --netG global_unet --resize_or_crop none \
--dataroot '/data/duan/sal_all/datasets/SARD' --batchSize 5 --data_csv 'train3.csv'

cd deep_saliency/Sal-CFS-GAN-master/
python3 train.py \
--name ar4 --no_instance --label_nc 0 --no_ganFeat_loss --netG global_unet --resize_or_crop none \
--dataroot '/data/duan/sal_all/datasets/SARD' --batchSize 5 --data_csv 'train4.csv'
```
- test Superimposed (dataroot, results_dir, how_many)
```
cd deep_saliency/Sal-CFS-GAN-master/
python3 test.py --name ar0 --netG global_unet --ngf 64 --resize_or_crop none --label_nc 0 --no_instance \
--dataroot '/data/duan/sal_all/datasets/SARD' --data_csv 'test0.csv' \
--results_dir './results/superimposed' \
--how_many 5000

cd deep_saliency/Sal-CFS-GAN-master/
python3 test.py --name ar1 --netG global_unet --ngf 64 --resize_or_crop none --label_nc 0 --no_instance \
--dataroot '/data/duan/sal_all/datasets/SARD' --data_csv 'test1.csv' \
--results_dir './results/superimposed' \
--how_many 5000

cd deep_saliency/Sal-CFS-GAN-master/
python3 test.py --name ar2 --netG global_unet --ngf 64 --resize_or_crop none --label_nc 0 --no_instance \
--dataroot '/data/duan/sal_all/datasets/SARD' --data_csv 'test2.csv' \
--results_dir './results/superimposed' \
--how_many 5000

cd deep_saliency/Sal-CFS-GAN-master/
python3 test.py --name ar3 --netG global_unet --ngf 64 --resize_or_crop none --label_nc 0 --no_instance \
--dataroot '/data/duan/sal_all/datasets/SARD' --data_csv 'test3.csv' \
--results_dir './results/superimposed' \
--how_many 5000

cd deep_saliency/Sal-CFS-GAN-master/
python3 test.py --name ar4 --netG global_unet --ngf 64 --resize_or_crop none --label_nc 0 --no_instance \
--dataroot '/data/duan/sal_all/datasets/SARD' --data_csv 'test4.csv' \
--results_dir './results/superimposed' \
--how_many 5000
```
- test AR (dataroot, results_dir, how_many)
```
cd deep_saliency/Sal-CFS-GAN-master/
python3 test.py --name label2city_512p --netG global_unet --ngf 64 --resize_or_crop none --label_nc 0 --no_instance \
--dataroot '/data/duan/sal_all/datasets/SARD/small_size/AR' \
--results_dir './results/AR' \
--how_many 5000
```
- test BG (dataroot, results_dir, how_many)
```
cd deep_saliency/Sal-CFS-GAN-master/
python3 test.py --name label2city_512p --netG global_unet --ngf 64 --resize_or_crop none --label_nc 0 --no_instance \
--dataroot '/data/duan/sal_all/datasets/SARD/small_size/BG' \
--results_dir './results/BG' \
--how_many 5000
```

**4.3 SALGAN** 
- train superimposed (flag, train_img_path, train_sal_path, val_img_path, val_sal_path, model_name)
```
cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python train_ar.py 'bce' \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'/data/duan/sal_all/datasets/SARD/small_size/fixMaps/' \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'/data/duan/sal_all/datasets/SARD/small_size/fixMaps/' \
'_ar0_bce' \
'/data/duan/sal_all/datasets/SARD/train0.csv' \
'/data/duan/sal_all/datasets/SARD/test0.csv'

cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python train_ar.py 'bce' \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'/data/duan/sal_all/datasets/SARD/small_size/fixMaps/' \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'/data/duan/sal_all/datasets/SARD/small_size/fixMaps/' \
'_ar1_bce' \
'/data/duan/sal_all/datasets/SARD/train1.csv' \
'/data/duan/sal_all/datasets/SARD/test1.csv'

cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python train_ar.py 'bce' \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'/data/duan/sal_all/datasets/SARD/small_size/fixMaps/' \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'/data/duan/sal_all/datasets/SARD/small_size/fixMaps/' \
'_ar2_bce' \
'/data/duan/sal_all/datasets/SARD/train2.csv' \
'/data/duan/sal_all/datasets/SARD/test2.csv'

cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python train_ar.py 'bce' \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'/data/duan/sal_all/datasets/SARD/small_size/fixMaps/' \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'/data/duan/sal_all/datasets/SARD/small_size/fixMaps/' \
'_ar3_bce' \
'/data/duan/sal_all/datasets/SARD/train3.csv' \
'/data/duan/sal_all/datasets/SARD/test3.csv'

cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python train_ar.py 'bce' \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'/data/duan/sal_all/datasets/SARD/small_size/fixMaps/' \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'/data/duan/sal_all/datasets/SARD/small_size/fixMaps/' \
'_ar4_bce' \
'/data/duan/sal_all/datasets/SARD/train4.csv' \
'/data/duan/sal_all/datasets/SARD/test4.csv'
```

- test superimposed (img_path, model_name, output_path)
```
cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python test_ar.py \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'_ar0_bce' \
'results/superimposed_bce' \
'/data/duan/sal_all/datasets/SARD/test0.csv'

cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python test_ar.py \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'_ar1_bce' \
'results/superimposed_bce' \
'/data/duan/sal_all/datasets/SARD/test1.csv'

cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python test_ar.py \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'_ar2_bce' \
'results/superimposed_bce' \
'/data/duan/sal_all/datasets/SARD/test2.csv'

cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python test_ar.py \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'_ar3_bce' \
'results/superimposed_bce' \
'/data/duan/sal_all/datasets/SARD/test3.csv'

cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python test_ar.py \
'/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' \
'_ar4_bce' \
'results/superimposed_bce' \
'/data/duan/sal_all/datasets/SARD/test4.csv'
```

- test AR (img_path, model_name, output_path)
```
cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python test.py \
'/data/duan/sal_all/datasets/SARD/small_size/AR/' \
'_salicon_bce' \
'results/AR_bce'
```

- test BG (img_path, model_name, output_path)
```
cd deep_saliency/salgan-master/scripts/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python test.py \
'/data/duan/sal_all/datasets/SARD/small_size/BG/' \
'_salicon_bce' \
'results/BG_bce'
```

**4.4 Salicon** 
- train superimposed
```
cd deep_saliency/Salicon_pytorch-master/src/
python3 train_ar.py \
--train_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--test_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--train_img_dir 'Superimposed' \
--train_label_dir 'fixMaps' \
--batch_size 20 \
--epochs 50 \
--model_name '_superimposed0' \
--train_data_csv '/data/duan/sal_all/datasets/SARD/train0.csv'

cd deep_saliency/Salicon_pytorch-master/src/
python3 train_ar.py \
--train_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--test_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--train_img_dir 'Superimposed' \
--train_label_dir 'fixMaps' \
--batch_size 20 \
--epochs 50 \
--model_name '_superimposed1' \
--train_data_csv '/data/duan/sal_all/datasets/SARD/train1.csv'

cd deep_saliency/Salicon_pytorch-master/src/
python3 train_ar.py \
--train_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--test_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--train_img_dir 'Superimposed' \
--train_label_dir 'fixMaps' \
--batch_size 20 \
--epochs 50 \
--model_name '_superimposed2' \
--train_data_csv '/data/duan/sal_all/datasets/SARD/train2.csv'

cd deep_saliency/Salicon_pytorch-master/src/
python3 train_ar.py \
--train_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--test_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--train_img_dir 'Superimposed' \
--train_label_dir 'fixMaps' \
--batch_size 20 \
--epochs 50 \
--model_name '_superimposed3' \
--train_data_csv '/data/duan/sal_all/datasets/SARD/train3.csv'

cd deep_saliency/Salicon_pytorch-master/src/
python3 train_ar.py \
--train_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--test_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--train_img_dir 'Superimposed' \
--train_label_dir 'fixMaps' \
--batch_size 20 \
--epochs 50 \
--model_name '_superimposed4' \
--train_data_csv '/data/duan/sal_all/datasets/SARD/train4.csv'
```

- test superimposed
```
cd deep_saliency/Salicon_pytorch-master/src/
python3 test_ar.py \
--test_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--test_img_dir 'Superimposed' \
--test_label_dir 'fixMaps' \
--model_name '_superimposed0' \
--test_data_csv '/data/duan/sal_all/datasets/SARD/test0.csv' \
--output_dir 'results_Superimposed'

cd deep_saliency/Salicon_pytorch-master/src/
python3 test_ar.py \
--test_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--test_img_dir 'Superimposed' \
--test_label_dir 'fixMaps' \
--model_name '_superimposed1' \
--test_data_csv '/data/duan/sal_all/datasets/SARD/test1.csv' \
--output_dir 'results_Superimposed'

cd deep_saliency/Salicon_pytorch-master/src/
python3 test_ar.py \
--test_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--test_img_dir 'Superimposed' \
--test_label_dir 'fixMaps' \
--model_name '_superimposed2' \
--test_data_csv '/data/duan/sal_all/datasets/SARD/test2.csv' \
--output_dir 'results_Superimposed'

cd deep_saliency/Salicon_pytorch-master/src/
python3 test_ar.py \
--test_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--test_img_dir 'Superimposed' \
--test_label_dir 'fixMaps' \
--model_name '_superimposed3' \
--test_data_csv '/data/duan/sal_all/datasets/SARD/test3.csv' \
--output_dir 'results_Superimposed'

cd deep_saliency/Salicon_pytorch-master/src/
python3 test_ar.py \
--test_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--test_img_dir 'Superimposed' \
--test_label_dir 'fixMaps' \
--model_name '_superimposed4' \
--test_data_csv '/data/duan/sal_all/datasets/SARD/test4.csv' \
--output_dir 'results_Superimposed'
```

- test ar
```
cd deep_saliency/Salicon_pytorch-master/src/
python3 test2.py \
--test_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--test_img_dir 'AR' \
--test_label_dir 'fixMaps' \
--model_name '' \
--output_dir 'results_AR'
```

- test bg
```
cd deep_saliency/Salicon_pytorch-master/src/
python3 test2.py \
--test_dataset_dir '/data/duan/sal_all/datasets/SARD/small_size' \
--test_img_dir 'BG' \
--test_label_dir 'fixMaps' \
--model_name '' \
--output_dir 'results_BG'
```

**4.5 SAM-VGG & SAM-ResNet** 

- train superimposed vgg (phase, model_dir)
```
cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 0 'png' 'sam-vgg_ar0' 'train0.csv' 'test0.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 0 'png' 'sam-vgg_ar1' 'train1.csv' 'test1.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 0 'png' 'sam-vgg_ar2' 'train2.csv' 'test2.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 0 'png' 'sam-vgg_ar3' 'train3.csv' 'test3.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 0 'png' 'sam-vgg_ar4' 'train4.csv' 'test4.csv'
```

- test superimposed vgg (phase, imgs_test_path, model_dir, output_folder)
```
cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test 0 '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'sam-vgg_ar0' 'results/Superimposed_vgg/' 'test0.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test 0 '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'sam-vgg_ar1' 'results/Superimposed_vgg/' 'test1.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test 0 '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'sam-vgg_ar2' 'results/Superimposed_vgg/' 'test2.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test 0 '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'sam-vgg_ar3' 'results/Superimposed_vgg/' 'test3.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test 0 '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'sam-vgg_ar4' 'results/Superimposed_vgg/' 'test4.csv'
```

- test AR (phase, imgs_test_path, model_dir, output_folder)
```
cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main.py test 0 '/data/duan/sal_all/datasets/SARD/small_size/AR/' 'sam-vgg_salicon' 'results/AR_vgg/'
```

- test BG (phase, imgs_test_path, model_dir, output_folder)
```
cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main.py test 0 '/data/duan/sal_all/datasets/SARD/small_size/BG/' 'sam-vgg_salicon' 'results/BG_vgg/'
```


- train superimposed resnet (phase, model_dir)
```
cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 1 'png' 'sam-resnet_ar0' 'train0.csv' 'test0.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 1 'png' 'sam-resnet_ar1' 'train1.csv' 'test1.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 1 'png' 'sam-resnet_ar2' 'train2.csv' 'test2.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 1 'png' 'sam-resnet_ar3' 'train3.csv' 'test3.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py train 1 'png' 'sam-resnet_ar4' 'train4.csv' 'test4.csv'
```

- test superimposed resnet (phase, imgs_test_path, model_dir, output_folder)
```
cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test 1 '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'sam-resnet_ar0' 'results/Superimposed_resnet/' 'test0.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test 1 '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'sam-resnet_ar1' 'results/Superimposed_resnet/' 'test1.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test 1 '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'sam-resnet_ar2' 'results/Superimposed_resnet/' 'test2.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test 1 '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'sam-resnet_ar3' 'results/Superimposed_resnet/' 'test3.csv'

cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main_ar.py test 1 '/data/duan/sal_all/datasets/SARD/small_size/Superimposed/' 'sam-resnet_ar4' 'results/Superimposed_resnet/' 'test4.csv'
```

- test AR (phase, imgs_test_path, model_dir, output_folder)
```
cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main.py test 1 '/data/duan/sal_all/datasets/SARD/small_size/AR/' 'sam-resnet_salicon' 'results/AR_resnet/'
```

- test BG (phase, imgs_test_path, model_dir, output_folder)
```
cd deep_saliency/sam-master/
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32,lib.cnmem=1,optimizer_including=cudnn \
python main.py test 1 '/data/duan/sal_all/datasets/SARD/small_size/BG/' 'sam-resnet_salicon' 'results/BG_resnet/'
```