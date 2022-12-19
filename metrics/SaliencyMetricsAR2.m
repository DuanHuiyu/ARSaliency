% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% calculate the saliency prediction performance on AR database
% alpha * AR + (1-alpha) * BG
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
saliencyMapsPath_AR = {'tra_results/AR/AIM/','tra_results/AR/CA/','tra_results/AR/CovSal/',...
    'tra_results/AR/GBVS/','tra_results/AR/HFT/','tra_results/AR/IT/',...
    'tra_results/AR/Judd/','tra_results/AR/Murray/','tra_results/AR/PFT/',...
    'tra_results/AR/SMVJ/','tra_results/AR/SR/','tra_results/AR/SUN/',...
    'tra_results/AR/SWD/'...
    };
saliencyMapsPath_BG = {'tra_results/BG/AIM/','tra_results/BG/CA/','tra_results/BG/CovSal/',...
    'tra_results/BG/GBVS/','tra_results/BG/HFT/','tra_results/BG/IT/',...
    'tra_results/BG/Judd/','tra_results/BG/Murray/','tra_results/BG/PFT/',...
    'tra_results/BG/SMVJ/','tra_results/BG/SR/','tra_results/BG/SUN/',...
    'tra_results/BG/SWD/'...
    };
saliencyMapsSuffix = {'.png','.png','.png',...
    '.png','.png','.png',...
    '.png','.png','.png',...
    '.png','.png','.png',...
    '.png',...
    };

alpha_list = [0.25,  0.5, 0.75];

shuffleMap = imread('AR_shuffle_map.png');

for cnt_model = 1:size(saliencyMapsPath_AR,2)
    clc
    cnt_model
    
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
        % img = imread([imgPath, img_list(cnt).name]);
        temp_name = split(img_list(cnt).name, '.');
        base_name = temp_name{1};
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
        temp_name = split(base_name, '_');
        base_name1 = [temp_name{3}, '_', temp_name{4}];
        base_name2 = [temp_name{1}, '_', temp_name{2}];
        alpha = alpha_list(uint8(str2double(temp_name{5}))+1);
        fixationPts = imread([fixationPtsPath, base_name, fixationPtsSuffix]);
        fixationMap = imread([fixationMapsPath, base_name, fixationMapsSuffix]);
        saliencyMap1 = imread([saliencyMapsPath_base, saliencyMapsPath_AR{cnt_model}, base_name1, saliencyMapsSuffix{cnt_model}]);
        saliencyMap2 = imread([saliencyMapsPath_base, saliencyMapsPath_BG{cnt_model}, base_name2, saliencyMapsSuffix{cnt_model}]);
        saliencyMap = alpha * saliencyMap1 + (1-alpha) * saliencyMap2;

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