% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% calculate all traditional saliency maps
% --------------------------------------------------------------------------

clc
clear all

% path_stimuli = '..\..\SARD\small_size\Superimposed\';
% path_write = '..\..\results\results_SARD\tra_results\Superimposed\';
% path_stimuli = '..\..\SARD\small_size\AR\';
% path_write = '..\..\results\results_SARD\results\tra_results\AR\';
% path_stimuli = '..\..\SARD\small_size\BG\';
% path_write = '..\..\results\results_SARD\results\tra_results\BG\';
% path_stimuli = '..\..\SARD\small_size\Superimposed_crop\';
% path_write = '..\..\results\results_SARD\tra_results\Superimposed_crop\';
% path_stimuli = '..\..\SARD\small_size\AR_crop\';
% path_write = '..\..\results\results_SARD\tra_results\AR_crop\';
path_stimuli = '..\..\SARD\small_size\BG_crop\';
path_write = '..\..\results\results_SARD\tra_results\BG_crop\';

path_tools = 'tra_saliency_src/';
addpath([path_tools,'Itti-1998/SaliencyToolbox'])   % IT: A model of saliency-based visual attention for rapid scene analysis
addpath([path_tools,'AIM/AIM']) % AIM: Saliency based on information maximization
addpath([path_tools,'GBVS/gbvs'])   % GBVS: Graph-based visual saliency
addpath([path_tools,'Hou-CVPR-SR']) % SR: Saliency detection: A spectral residual approach
addpath([path_tools,'SUN/imagesaliency/saliency']) % SUN: SUN: a Bayesian framework for saliency using natural statistics
addpath([path_tools,'PQFT'])% PFT: Spatio-temporal saliency detection using phase spectrum of quaternion fourier transform
addpath(genpath('tra_saliency_lib/SMVJ'))	% SMVJ: Predicting human gaze using low-level saliency combined with face detection
addpath(genpath('tra_saliency_lib/JuddSaliency'))    % Judd: Learning to predict where humans look
addpath([path_tools,'SWD/Wu_Saliency_CVPR11'])    % SWD: Visual saliency detection by spatially weighted dissimilarity
addpath([path_tools,'Murray/SIM'])	% Murray: Saliency estimation using a non-parametric low-level vision model
addpath([path_tools,'ContextAware/Saliency'])  % CA: Context-aware saliency detection
% addpath(genpath([path_tools,'BMS/BMS_v2-mex']))  % BMS: Saliency detection: a Boolean map approach
addpath([path_tools,'CovSal/saliency']) % CovSal: Visual saliency estimation by nonlinearly integrating features using region covariances
addpath(genpath([path_tools,'HFT/HFT Code'])) % HFT: Visual saliency based on scale-space analysis in the frequency domain

addpath(genpath('tra_saliency_lib/matlabPyrTools'));
addpath(genpath('tra_saliency_lib/LabelMeToolbox-master'));
addpath(genpath('tra_saliency_lib/Felzenszwalb_car&person'));
% addpath(genpath('lib'));

cd tra_saliency_lib/gbvs
gbvs_install
cd ../..

ALG = {'IT'; 'AIM'; 'GBVS'; 'SR'; 'SUN'; 'PFT'; 'SMVJ'; 'Judd'; 'SWD'; 'Murray'; 'CA'; 'CovSal'; 'HFT'; 'BMS'}';

% prepare all stimuli
ext = {'*.jpeg','*.jpg','*.png'};
images = [];
for cnt = 1:length(ext)
    images = [images; dir([path_stimuli, ext{cnt}])];
end

% prepare write folders
for cnt = 1:length(ALG)
    path_write_ALG{cnt} = [path_write, ALG{cnt}, '\'];
    if ~exist(path_write_ALG{cnt})
        mkdir(path_write_ALG{cnt})
    end
end

% compute all saliency
for cnt = 1:length(images)
    clc
    cnt
    
    img_name = images(cnt).name;
    img_path = [path_stimuli, img_name];
    img = imread(img_path);
    img_double = double(img);
    img_size = size(img);

    % --- compute IT --- 
    img_IT = initializeImage(img_path);
    params = defaultSaliencyParams;
    sal_IT = makeSaliencyMap(img_IT,params);
    sal_IT = imresize(sal_IT.data,img_IT.size(1:2));
    imwrite(sal_IT, [path_write_ALG{1}, img_name]);
    % --- compute AIM --- 
    if mean(mean(mean(img))) == 0   % AIM cannot work for zero image
        sal_AIM = sal_IT;
    else
        sal_AIM = AIM(img_path);
        sal_AIM = sal_AIM - min(min(sal_AIM));
        sal_AIM = sal_AIM./max(max(sal_AIM));
    end
    imwrite(sal_AIM, [path_write_ALG{2}, img_name]);
    % --- compute GBVS ---
    sal_gbvs = gbvs(img_path);
    imwrite(sal_gbvs.master_map_resized, [path_write_ALG{3}, img_name]);
    % --- compute SR ---
    sal_SR = SR(img);
    imwrite(sal_SR, [path_write_ALG{4}, img_name]);
    % --- compute SUN ---
    sal_SUN = saliencyimage_convolution(img_double,0.5);
    sal_SUN = sal_SUN - min(min(sal_SUN));
    sal_SUN = sal_SUN./max(max(sal_SUN));
    sal_SUN = imresize(sal_SUN, img_size(1:2));
    imwrite(sal_SUN, [path_write_ALG{5}, img_name]);
    % --- compute PFT ---
    sal_PFT = PFT(img);
    imwrite(sal_PFT, [path_write_ALG{6}, img_name]);
    % --- compute SMVJ ---
    sal_SMVJ = SMVJ_Main(img_path); % need 32-bit matlat to work
    imwrite(sal_SMVJ.master_map_resized, [path_write_ALG{7}, img_name]);
    % --- compute Judd ---
    if mean(mean(mean(img))) == 0   % Judd cannot work for zero image
        sal_Judd = sal_IT;
    else
        sal_Judd = JuddSaliency(img_path);
    end
    imwrite(sal_Judd, [path_write_ALG{8}, img_name]);
    % --- compute SWD ---
    winSize = 14; %size of each patch: 14x14
    rdDim = 11; %reducing to 11 dimensions
    sigma = 3; %sigma of Gaussian used for smoothing the saliency map
    sal_SWD = Wu_ImageSaliencyComputing(img_path, winSize, rdDim, sigma);
    imwrite(sal_SWD, [path_write_ALG{9}, img_name]);
    % --- compute Murray ---
    [m n p]      = size(img_double);
    window_sizes = [13 26];                          % window sizes for computing center-surround contrast
    wlev         = min([7,floor(log2(min([m n])))]); % number of wavelet planes
    gamma        = 2.4;                              % gamma value for gamma correction
    srgb_flag    = 1;                                % 0 if img is rgb; 1 if img is srgb
    sal_Murray = SIM(img_double, window_sizes, wlev, gamma, srgb_flag);
    sal_Murray = sal_Murray - min(min(sal_Murray));
    sal_Murray = sal_Murray./max(max(sal_Murray));
    imwrite(sal_Murray, [path_write_ALG{10}, img_name]);
    % --- compute CA ---
    file_names{1} = img_path;
    sal_CA = ca_saliency(file_names);
    sal_CA = imresize(sal_CA{1}.SaliencyMap, img_size(1:2));
    imwrite(sal_CA, [path_write_ALG{11}, img_name]);
    rmdir([path_stimuli,'Output'], 's') % CA will generate a 'Output' folder in the stimuli folder, we remove it here
    % --- compute CovSal ---
    options.size = 512;                     % size of rescaled image
    options.quantile = 1/10;                % parameter specifying the most similar regions in the neighborhood
    options.centerBias = 1;                 % 1 for center bias and 0 for no center bias
    options.modeltype = 'CovariancesOnly';  % 'CovariancesOnly' and 'SigmaPoints' to denote whether first-order statistics will be incorporated or not
    sal_CovSal = saliencymap(img_path, options);
    sal_CovSal = sal_CovSal - min(min(sal_CovSal));
    sal_CovSal = sal_CovSal./max(max(sal_CovSal));
    imwrite(sal_CovSal, [path_write_ALG{12}, img_name]);
    % --- compute HFT ---
    sal_HFT = HFT(img_double);
    imwrite(sal_HFT, [path_write_ALG{13}, img_name]);
end

% --- compute BMS --- (not working)
% BMS(path_stimuli,path_write_ALG{14});


