## Saliency Models

This folder contains all baseline saliency models and the proposed VQSal & VQSal-AR model.

### Baseline Saliency Models

Please follow the instructions in the `baseline_saliency_models` to test all baseline models.

### VQSal & VQSal-AR

Please follow the instructions in the `VQSal` to train/test the proposed models.

### Calculate the three types of model proposed in the paper

- Generate train/test split files:
```
write_train_test_split_csv.m
```
- Generate Type III traditional saliency maps:
```
train_AR_traditional.m
```
- Generate Type III deep saliency maps:
```
train_AR_deep.m
```