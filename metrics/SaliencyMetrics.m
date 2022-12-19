% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% calculate the saliency prediction performance
% run AUC_Judd, AUC_shuffled, NSS, CC
% --------------------------------------------------------------------------

clc
clear all

addpath('code_forMetrics\')

%% path
imgPath = '..\Salicon\val\images\';
fixationPtsPath = '..\Salicon\val\fixpts_map\';
fixationPtsSuffix = '.png';
fixationMapsPath = '..\Salicon\val\maps\';
fixationMapsSuffix = '.png';
img_list=dir([imgPath, '*.jpg']);

saliencyMapsPath = '..\results\salicon\salicon\';
saliencyMapsSuffix = '.png';
% saliencyMapsPath = '..\results\salicon\salicon4\';
% saliencyMapsSuffix = '.png';
% saliencyMapsPath = '..\results\salicon\mlnet\';
% saliencyMapsSuffix = '.jpg';
% saliencyMapsPath = '..\results\salicon\sam_vgg\';
% saliencyMapsSuffix = '.jpg';
% saliencyMapsPath = '..\results\salicon\sam_resnet\';
% saliencyMapsSuffix = '.jpg';
% saliencyMapsPath = '..\results\salicon\salgan_bce\';
% saliencyMapsSuffix = '.png';
% saliencyMapsPath = '..\results\salicon\salgan_bce50\';
% saliencyMapsSuffix = '.png';
% saliencyMapsPath = '..\results\salicon\salgan\';
% saliencyMapsSuffix = '.png';
% saliencyMapsPath = '..\results\salicon\GazeGAN\';
% saliencyMapsSuffix = '_synthesized_image.jpg';


shuffleMap = imread('Salicon_shuffle_map.png');

AUC_Borji_score_all = []; 
AUC_Judd_score_all = [];
sAUC_score_all = [];
CC_score_all = [];
IG_score_all = [];
KL_score_all = [];
NSS_score_all = [];
SIM_score_all = [];

for cnt = 1:length(img_list)
    cnt
    % img = imread([imgPath, img_list(cnt).name]);
    temp_name = split(img_list(cnt).name, '.');
    base_name = temp_name{1};
    fixationPts = imread([fixationPtsPath, base_name, fixationPtsSuffix]);
    fixationMap = imread([fixationMapsPath, base_name, fixationMapsSuffix]);
    saliencyMap = imread([saliencyMapsPath, base_name, saliencyMapsSuffix]);
    
    if size(saliencyMap,3)>1
        saliencyMap = rgb2gray(saliencyMap);
        % saliencyMap = mean(saliencyMap,3);
    end
    baselineMap = logical(imresize(shuffleMap, size(fixationPts), 'nearest')) & (fixationPts==0);
    
    % compute performance
    [AUC_Borji_score, AUC_Judd_score, sAUC_score, CC_score, IG_score, KL_score, NSS_score, SIM_score] ...
        = run_all_metrics(saliencyMap, fixationMap, fixationPts, baselineMap);
    
    % append all
    AUC_Borji_score_all(cnt) = AUC_Borji_score;
    AUC_Judd_score_all(cnt) = AUC_Judd_score;
    sAUC_score_all(cnt) = sAUC_score;
    CC_score_all(cnt) = CC_score;
    IG_score_all(cnt) = IG_score;
    KL_score_all(cnt) = KL_score;
    NSS_score_all(cnt) = NSS_score;
    SIM_score_all(cnt) = SIM_score;
end

% average
avg_AUC_Borji_score = mean(AUC_Borji_score_all);
avg_AUC_Judd_score = mean(AUC_Judd_score_all);
avg_AUC_shuffle_score = mean(sAUC_score_all);
avg_CC_score = mean(CC_score_all);
avg_IG_score = mean(IG_score_all);
avg_KL_score = mean(KL_score_all);
avg_NSS_score = mean(NSS_score_all);
avg_SIM_score = mean(SIM_score_all);