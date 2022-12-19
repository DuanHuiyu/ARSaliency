## VQSal & VQSal-AR

### 0. Pretrained Weights and Models
All pretrained weights and models can be download here: [[百度网盘](https://pan.baidu.com/s/1thXJaTT_weEv79_8tKS8tw?pwd=ihtg)] [[TeraBox](https://terabox.com/s/1NAc5XMOi1n6WH_S62E9MVQ)].

### 1. Installation (PyTorch)
- create environment
```
conda create -n saliency python=3.8 pip
conda activate saliency
```
- for CUDA 11.3 users
```
conda install pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=11.3 -c pytorch
```
- for CUDA 10.2 users
``` 
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=10.2 -c pytorch
```
- install other requirements
```
pip install -r requirements.txt
pip install git+https://github.com/openai/DALL-E.git &> /dev/null   # for inference
```

### 2. Train/Test `VQSal` on Salicon
- Train Salicon (need to change the data path in `sal_configs/saliency_salicon.yaml`):
```
python main_transfer.py --base sal_configs/saliency_salicon.yaml -t True --gpus 0,1,2,3
```
- Test Salicon (need to change the data path in `sal_configs/saliency_salicon.yaml`):
```
python main_test.py \
--config_path 'sal_configs/saliency_salicon.yaml' \
--model_path 'logs/saliency_salicon/checkpoints/last.ckpt' \
--input_path '/DATA/DATA1/Saliency/Salicon/val/images' \
--output_path 'results/salicon'
```

### 3. Test `VQSal` with Salicon Weights on SARD (prepare for Type II and Type III)
- test AR
```
python main_test.py \
--config_path 'sal_configs/saliency_salicon.yaml' \
--model_path 'logs/saliency_salicon/checkpoints/last.ckpt' \
--input_path '/DATA/DATA1/Saliency/SARD/small_size/AR' \
--output_path 'results/AR'
```
- test BG
```
python main_test.py \
--config_path 'sal_configs/saliency_salicon.yaml' \
--model_path 'logs/saliency_salicon/checkpoints/last.ckpt' \
--input_path '/DATA/DATA1/Saliency/SARD/small_size/BG' \
--output_path 'results/BG'
```

### 3. Train/Test `VQSal` on SARD (Type I)
- train AR baseline (VQSal: Type I)
```
python main_transfer.py --base sal_configs/saliency_AR_base.yaml -t True --gpus 0,
python main_transfer.py --base sal_configs/saliency_AR_base1.yaml -t True --gpus 0,
python main_transfer.py --base sal_configs/saliency_AR_base2.yaml -t True --gpus 0,
python main_transfer.py --base sal_configs/saliency_AR_base3.yaml -t True --gpus 0,
python main_transfer.py --base sal_configs/saliency_AR_base4.yaml -t True --gpus 0,
```
- test AR baseline (VQSal: Type I)
```
python main_test_baseline.py \
--config_path 'sal_configs/saliency_AR_base.yaml' \
--model_path 'logs/saliency_AR_base/checkpoints/last.ckpt' \
--input_path '/DATA/DATA1/Saliency/SARD/small_size' --output_path 'results/Superimposed_base' \
--data_csv '/DATA/DATA1/Saliency/SARD/test0.csv'

python main_test_baseline.py \
--config_path 'sal_configs/saliency_AR_base1.yaml' \
--model_path 'logs/saliency_AR_base1/checkpoints/last.ckpt' \
--input_path '/DATA/DATA1/Saliency/SARD/small_size' --output_path 'results/Superimposed_base' \
--data_csv '/DATA/DATA1/Saliency/SARD/test1.csv'

python main_test_baseline.py \
--config_path 'sal_configs/saliency_AR_base2.yaml' \
--model_path 'logs/saliency_AR_base2/checkpoints/last.ckpt' \
--input_path '/DATA/DATA1/Saliency/SARD/small_size' --output_path 'results/Superimposed_base' \
--data_csv '/DATA/DATA1/Saliency/SARD/test2.csv'

python main_test_baseline.py \
--config_path 'sal_configs/saliency_AR_base3.yaml' \
--model_path 'logs/saliency_AR_base3/checkpoints/last.ckpt' \
--input_path '/DATA/DATA1/Saliency/SARD/small_size' --output_path 'results/Superimposed_base' \
--data_csv '/DATA/DATA1/Saliency/SARD/test3.csv'

python main_test_baseline.py \
--config_path 'sal_configs/saliency_AR_base4.yaml' \
--model_path 'logs/saliency_AR_base4/checkpoints/last.ckpt' \
--input_path '/DATA/DATA1/Saliency/SARD/small_size' --output_path 'results/Superimposed_base' \
--data_csv '/DATA/DATA1/Saliency/SARD/test4.csv'
```

### 4. Train/Test `VQSal-AR` on SARD
- train AR saliency (VQSal-AR)
```
python main_transfer.py --base sal_configs/saliency_AR.yaml -t True --gpus 0,1
python main_transfer.py --base sal_configs/saliency_AR1.yaml -t True --gpus 0,1
python main_transfer.py --base sal_configs/saliency_AR2.yaml -t True --gpus 0,1
python main_transfer.py --base sal_configs/saliency_AR3.yaml -t True --gpus 0,1
python main_transfer.py --base sal_configs/saliency_AR4.yaml -t True --gpus 0,1
```

- test AR saliency (VQSal-AR)
```
python main_test_ar.py \
--config_path 'sal_configs/saliency_AR.yaml' \
--model_path 'logs/saliency_AR/checkpoints/last.ckpt' \
--input_path '/DATA/DATA1/Saliency/SARD/small_size' \
--output_path 'results/Superimposed3' \
--data_csv '/DATA/DATA1/Saliency/SARD/test0.csv'

python main_test_ar.py \
--config_path 'sal_configs/saliency_AR.yaml' \
--model_path 'logs/saliency_AR1/checkpoints/last.ckpt' \
--input_path '/DATA/DATA1/Saliency/SARD/small_size' \
--output_path 'results/Superimposed3' \
--data_csv '/DATA/DATA1/Saliency/SARD/test1.csv'

python main_test_ar.py \
--config_path 'sal_configs/saliency_AR.yaml' \
--model_path 'logs/saliency_AR2/checkpoints/last.ckpt' \
--input_path '/DATA/DATA1/Saliency/SARD/small_size' \
--output_path 'results/Superimposed3' \
--data_csv '/DATA/DATA1/Saliency/SARD/test2.csv'

python main_test_ar.py \
--config_path 'sal_configs/saliency_AR.yaml' \
--model_path 'logs/saliency_AR3/checkpoints/last.ckpt' \
--input_path '/DATA/DATA1/Saliency/SARD/small_size' \
--output_path 'results/Superimposed3' \
--data_csv '/DATA/DATA1/Saliency/SARD/test3.csv'

python main_test_ar.py \
--config_path 'sal_configs/saliency_AR.yaml' \
--model_path 'logs/saliency_AR4/checkpoints/last.ckpt' \
--input_path '/DATA/DATA1/Saliency/SARD/small_size' \
--output_path 'results/Superimposed3' \
--data_csv '/DATA/DATA1/Saliency/SARD/test4.csv'
```