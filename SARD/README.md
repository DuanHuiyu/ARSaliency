## SARD

The SARD database consists of 450 AR images 450 BG images and 1350 Superimposed images.

### Download Database
1. To download the SARD database (small size): [[百度网盘](https://pan.baidu.com/s/1sB79k_dCSSv90eFjiWv41A?pwd=2ri7)] [[TeraBox](https://terabox.com/s/1IJlxQ_o5ZGSi8aiMrJdRfA)].
It should be noted that this link is a small size of SARD, but we use the small size dataset for evaluating all models.

2. To download the SARD database (full size): [[百度网盘](https://pan.baidu.com/s/1ukXKBGJcDXxQq9AbVR-iEQ?pwd=te2q)].
The full size SARD includes all images with full sizes. This folder also contains heat maps we used in the paper.

3. If you would like to generate all stimuli for subjective experiments, please download the raw images here [[百度网盘](https://pan.baidu.com/s/1ClF_dNNust2dj0FDlAlUrw?pwd=e15h)] [[TeraBox](https://terabox.com/s/1MBczl6kAUt8jgmz1AsWEuQ)], and then run the preprocess codes in "..\subjective_experiments" folder.

### Description
+ Folder naming convention for small size SARD:

        .\small_size\AR: All AR images with perceptual FOV
        .\small_size\BG: All BG images with perceptual FOV
        .\small_size\Superimposed: All Superimposed images with perceptual FOV
        .\small_size\fixPts: All fixPts images with perceptual FOV
        .\small_size\fixMaps: All fixMaps images with perceptual FOV
        .\small_size\AR: All AR images with AR FOV
        .\small_size\BG: All BG images with AR FOV
        .\small_size\Superimposed: All Superimposed images with AR FOV
        .\small_size\fixPts: All fixPts images with AR FOV
        .\small_size\fixMaps: All fixMaps images with AR FOV

+ File naming convention for Superimposed, fixPts, and fixMaps images:

        BG img Name + AR img Name + mixing level

### Citation
If you use the CFIQA database, please consider citing:

    @inproceedings{duan2022saliency,
        title={Saliency in Augmented Reality},
        author={Duan, Huiyu and Shen, Wei and Min, Xiongkuo and Tu, Danyang and Li, Jing and Zhai, Guangtao},
        booktitle={Proceedings of the ACM International Conference on Multimedia (ACM MM)},
        year={2022}
    }

### Contact
If you have any question, please contact huiyuduan@sjtu.edu.cn