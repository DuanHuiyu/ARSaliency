% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% calculate the saliency prediction performance on AR database
% only on superimposed images
% --------------------------------------------------------------------------

clc
clear all

addpath('code_forMetrics\')

%% path
imgPath = '..\SARD\small_size\Superimposed\';
fixationPtsPath = '..\SARD\small_size\fixPts\';
fixationPtsSuffix = '.png';
fixationMapsPath = '..\SARD\small_size\fixMaps\';
fixationMapsSuffix = '.png';
img_list=dir([imgPath, '*.png']);

saliencyMapsPath_base = '..\results\results_SARD\';
saliencyMapsPath = {'deep_results/Superimposed/Salicon/','deep_results/Superimposed/mlnet/',...
    'deep_results/Superimposed/sam_vgg/','deep_results/Superimposed/sam_resnet/',...
    'deep_results/Superimposed/salgan/','deep_results/Superimposed/gazegan/',...
    'deep_results/Superimposed/VQSal/'
    };
saliencyMapsSuffix = {'.png','.png',...
    '.png','.png',...
    '.png','_synthesized_image.jpg',...
    '.png'
    };

shuffleMap = imread('AR_shuffle_map.png');

for cnt_model = 1:size(saliencyMapsPath,2)
    clc
    cnt_model
    
    % since trainable split may not cover all images we use predicted images as base names
    img_list=dir([saliencyMapsPath_base, saliencyMapsPath{cnt_model}, '*', saliencyMapsSuffix{cnt_model}]);
    
    AUC_Borji_score_all{cnt_model} = []; 
    AUC_Judd_score_all{cnt_model} = [];
    sAUC_score_all{cnt_model} = [];
    CC_score_all{cnt_model} = [];
    IG_score_all{cnt_model} = [];
    KL_score_all{cnt_model} = [];
    NSS_score_all{cnt_model} = [];
    SIM_score_all{cnt_model} = [];
    
    for cnt = 1:length(img_list)
        cnt
%         temp_name = split(img_list(cnt).name, '.');
%         base_name = temp_name{1};
        base_name = erase(img_list(cnt).name, saliencyMapsSuffix{cnt_model});
        % --- below for specific category ---
        temp_name2 = split(base_name, '_');
        if strcmp(temp_name2{4},'graphic')  % choose from: "graphic", "natural", "webpage"
            category(cnt)=0;
        elseif strcmp(temp_name2{4},'natural')
            category(cnt)=1;
        elseif strcmp(temp_name2{4},'webpage')
            category(cnt)=2;
        end
        % -----------------------------------
        fixationPts = imread([fixationPtsPath, base_name, fixationPtsSuffix]);
        fixationMap = imread([fixationMapsPath, base_name, fixationMapsSuffix]);
        saliencyMap = imread([saliencyMapsPath_base, saliencyMapsPath{cnt_model}, base_name, saliencyMapsSuffix{cnt_model}]);

        if size(saliencyMap,3)>1
            saliencyMap = rgb2gray(saliencyMap);
            % saliencyMap = mean(saliencyMap,3);
        end
        baselineMap = logical(imresize(shuffleMap, size(fixationPts), 'nearest')) & (fixationPts==0);

        % compute performance
        [AUC_Borji_score, AUC_Judd_score, sAUC_score, CC_score, IG_score, KL_score, NSS_score, SIM_score] ...
            = run_all_metrics(saliencyMap, fixationMap, fixationPts, baselineMap);

        % append all
        AUC_Borji_score_all{cnt_model}(cnt) = AUC_Borji_score;
        AUC_Judd_score_all{cnt_model}(cnt) = AUC_Judd_score;
        sAUC_score_all{cnt_model}(cnt) = sAUC_score;
        CC_score_all{cnt_model}(cnt) = CC_score;
        IG_score_all{cnt_model}(cnt) = IG_score;
        KL_score_all{cnt_model}(cnt) = KL_score;
        NSS_score_all{cnt_model}(cnt) = NSS_score;
        SIM_score_all{cnt_model}(cnt) = SIM_score;
    end

    % average
    avg_AUC_Borji_score(1,cnt_model) = mean(AUC_Borji_score_all{cnt_model});
    avg_AUC_Judd_score(1,cnt_model) = mean(AUC_Judd_score_all{cnt_model});
    avg_AUC_shuffle_score(1,cnt_model) = mean(sAUC_score_all{cnt_model});
    avg_CC_score(1,cnt_model) = mean(CC_score_all{cnt_model});
    avg_IG_score(1,cnt_model) = mean(IG_score_all{cnt_model});
    avg_KL_score(1,cnt_model) = mean(KL_score_all{cnt_model});
    avg_NSS_score(1,cnt_model) = mean(NSS_score_all{cnt_model});
    avg_SIM_score(1,cnt_model) = mean(SIM_score_all{cnt_model});
    
    
    % --- below for specific category ---
    avg_AUC_Borji_score(2,cnt_model) = mean(AUC_Borji_score_all{cnt_model}(find(category==0)));
    avg_AUC_Judd_score(2,cnt_model) = mean(AUC_Judd_score_all{cnt_model}(find(category==0)));
    avg_AUC_shuffle_score(2,cnt_model) = mean(sAUC_score_all{cnt_model}(find(category==0)));
    avg_CC_score(2,cnt_model) = mean(CC_score_all{cnt_model}(find(category==0)));
    avg_IG_score(2,cnt_model) = mean(IG_score_all{cnt_model}(find(category==0)));
    avg_KL_score(2,cnt_model) = mean(KL_score_all{cnt_model}(find(category==0)));
    avg_NSS_score(2,cnt_model) = mean(NSS_score_all{cnt_model}(find(category==0)));
    avg_SIM_score(2,cnt_model) = mean(SIM_score_all{cnt_model}(find(category==0)));
    
    avg_AUC_Borji_score(3,cnt_model) = mean(AUC_Borji_score_all{cnt_model}(find(category==1)));
    avg_AUC_Judd_score(3,cnt_model) = mean(AUC_Judd_score_all{cnt_model}(find(category==1)));
    avg_AUC_shuffle_score(3,cnt_model) = mean(sAUC_score_all{cnt_model}(find(category==1)));
    avg_CC_score(3,cnt_model) = mean(CC_score_all{cnt_model}(find(category==1)));
    avg_IG_score(3,cnt_model) = mean(IG_score_all{cnt_model}(find(category==1)));
    avg_KL_score(3,cnt_model) = mean(KL_score_all{cnt_model}(find(category==1)));
    avg_NSS_score(3,cnt_model) = mean(NSS_score_all{cnt_model}(find(category==1)));
    avg_SIM_score(3,cnt_model) = mean(SIM_score_all{cnt_model}(find(category==1)));
    
    avg_AUC_Borji_score(4,cnt_model) = mean(AUC_Borji_score_all{cnt_model}(find(category==2)));
    avg_AUC_Judd_score(4,cnt_model) = mean(AUC_Judd_score_all{cnt_model}(find(category==2)));
    avg_AUC_shuffle_score(4,cnt_model) = mean(sAUC_score_all{cnt_model}(find(category==2)));
    avg_CC_score(4,cnt_model) = mean(CC_score_all{cnt_model}(find(category==2)));
    avg_IG_score(4,cnt_model) = mean(IG_score_all{cnt_model}(find(category==2)));
    avg_KL_score(4,cnt_model) = mean(KL_score_all{cnt_model}(find(category==2)));
    avg_NSS_score(4,cnt_model) = mean(NSS_score_all{cnt_model}(find(category==2)));
    avg_SIM_score(4,cnt_model) = mean(SIM_score_all{cnt_model}(find(category==2)));
    % -----------------------------------
end