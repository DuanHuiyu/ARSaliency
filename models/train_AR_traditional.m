% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% SVR train/test
% --------------------------------------------------------------------------

clc
clear all

fprintf('Setting up the environment.\n');

% libsvm
addpath('baseline_saliency_models/tra_saliency_lib/libsvm');

% liblinear
addpath('baseline_saliency_models/tra_saliency_lib/liblinear');

Features{1} = {'AR/AIM', 'BG/AIM', 'Superimposed/AIM'};
FeatureSuffix{1} = {'.png', '.png', '.png'};
Features{2} = {'AR/CA', 'BG/CA', 'Superimposed/CA'};
FeatureSuffix{2} = {'.png', '.png', '.png'};
Features{3} = {'AR/CovSal', 'BG/CovSal', 'Superimposed/CovSal'};
FeatureSuffix{3} = {'.png', '.png', '.png'};
Features{4} = {'AR/GBVS', 'BG/GBVS', 'Superimposed/GBVS'};
FeatureSuffix{4} = {'.png', '.png', '.png'};
Features{5} = {'AR/HFT', 'BG/HFT', 'Superimposed/HFT'};
FeatureSuffix{5} = {'.png', '.png', '.png'};
Features{6} = {'AR/IT', 'BG/IT', 'Superimposed/IT'};
FeatureSuffix{6} = {'.png', '.png', '.png'};
Features{7} = {'AR/Judd', 'BG/Judd', 'Superimposed/Judd'};
FeatureSuffix{7} = {'.png', '.png', '.png'};
Features{8} = {'AR/Murray', 'BG/Murray', 'Superimposed/Murray'};
FeatureSuffix{8} = {'.png', '.png', '.png'};
Features{9} = {'AR/PFT', 'BG/PFT', 'Superimposed/PFT'};
FeatureSuffix{9} = {'.png', '.png', '.png'};
Features{10} = {'AR/SMVJ', 'BG/SMVJ', 'Superimposed/SMVJ'};
FeatureSuffix{10} = {'.png', '.png', '.png'};
Features{11} = {'AR/SR', 'BG/SR', 'Superimposed/SR'};
FeatureSuffix{11} = {'.png', '.png', '.png'};
Features{12} = {'AR/SUN', 'BG/SUN', 'Superimposed/SUN'};
FeatureSuffix{12} = {'.png', '.png', '.png'};
Features{13} = {'AR/SWD', 'BG/SWD', 'Superimposed/SWD'};
FeatureSuffix{13} = {'.png', '.png', '.png'};

OutputName{1} = 'AIM/';
OutputName{2} = 'CA/';
OutputName{3} = 'CovSal/';
OutputName{4} = 'GBVS/';
OutputName{5} = 'HFT/';
OutputName{6} = 'IT/';
OutputName{7} = 'Judd/';
OutputName{8} = 'Murray/';
OutputName{9} = 'PFT/';
OutputName{10} = 'SMVJ/';
OutputName{11} = 'SR/';
OutputName{12} = 'SUN/';
OutputName{13} = 'SWD/';

splits = 5;

% parameters
csvPath = 'train_test_split\';
imgPath = '..\SARD\small_size\Superimposed\';
fixationPtsPath = '..\SARD\small_size\fixPts\';
fixationPtsSuffix = '.png';
fixationMapsPath = '..\SARD\small_size\fixMaps\';
fixationMapsSuffix = '.png';
featurePath = '..\results\results_SARD\tra_results\';
img_list=dir([imgPath, '*.png']);

outputPath = '..\results\results_SARD\tra_results_svm\';

for m = 1:length(Features)
    clc
    m
    
    for current_split = 1:splits
        current_split
        train_file = importdata([csvPath, 'train',num2str(current_split-1),'.csv']);
        for cnt_img = 2:length(train_file)
            temp_name = split(train_file{cnt_img}, ',');
            trainingImgs{cnt_img-1,1} = temp_name{2};   % AR
            trainingImgs{cnt_img-1,2} = temp_name{1};   % BG
            trainingImgs{cnt_img-1,3} = temp_name{3};   % BG
        end
        test_file = importdata([csvPath, 'test',num2str(current_split-1),'.csv']);
        for cnt_img = 2:length(test_file)
            temp_name = split(test_file{cnt_img}, ',');
            testImgs{cnt_img-1,1} = temp_name{2};   % AR
            testImgs{cnt_img-1,2} = temp_name{1};   % BG
            testImgs{cnt_img-1,3} = temp_name{3};   % BG
        end
        
        % prepare data
        [X Y] = train_sampling(fixationMapsPath, featurePath, trainingImgs, Features{m}, FeatureSuffix{m});
        % train svm
        model = train_liblinear(X, Y);
        % test and save saliency images in current split
        train_test(model, '', featurePath, testImgs, Features{m}, FeatureSuffix{m}, [outputPath, OutputName{m}]);
    end
    
end