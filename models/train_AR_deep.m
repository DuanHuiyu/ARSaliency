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

Features{1} = {'AR/Salicon', 'BG/Salicon', 'Superimposed/Salicon'};
FeatureSuffix{1} = {'.png', '.png', '.png'};
Features{2} = {'AR/mlnet', 'BG/mlnet', 'Superimposed/mlnet'};
FeatureSuffix{2} = {'.png', '.png', '.png'};
Features{3} = {'AR/sam_vgg', 'BG/sam_vgg', 'Superimposed/sam_vgg'};
FeatureSuffix{3} = {'.png', '.png', '.png'};
Features{4} = {'AR/sam_resnet', 'BG/sam_resnet', 'Superimposed/sam_resnet'};
FeatureSuffix{4} = {'.png', '.png', '.png'};
Features{5} = {'AR/salgan', 'BG/salgan', 'Superimposed/salgan'};
FeatureSuffix{5} = {'.png', '.png', '.png'};
Features{6} = {'AR/gazegan', 'BG/gazegan', 'Superimposed/gazegan'};
FeatureSuffix{6} = {'_synthesized_image.jpg', '_synthesized_image.jpg', '_synthesized_image.jpg'};
Features{7} = {'AR/VQSal', 'BG/VQSal', 'Superimposed/VQSal'};
FeatureSuffix{7} = {'.png', '.png', '.png'};

OutputName{1} = 'Salicon/';
OutputName{2} = 'mlnet/';
OutputName{3} = 'sam_vgg/';
OutputName{4} = 'sam_resnet/';
OutputName{5} = 'salgan/';
OutputName{6} = 'gazegan/';
OutputName{7} = 'VQSal/';

splits = 5;

% parameters
csvPath = 'train_test_split\';
imgPath = '..\SARD\small_size\Superimposed\';
fixationPtsPath = '..\SARD\small_size\fixPts\';
fixationPtsSuffix = '.png';
fixationMapsPath = '..\SARD\small_size\fixMaps\';
fixationMapsSuffix = '.png';
featurePath = '..\results\results_SARD\deep_results\';
img_list=dir([imgPath, '*.png']);

outputPath = '..\results\results_SARD\deep_results_3\';

for m = 1:length(Features)
    clc
    m
    
    for current_split = 1:splits
        current_split
        train_file = importdata(['train',num2str(current_split-1),'.csv']);
        for cnt_img = 2:length(train_file)
            temp_name = split(train_file{cnt_img}, ',');
            trainingImgs{cnt_img-1,1} = temp_name{2};   % AR
            trainingImgs{cnt_img-1,2} = temp_name{1};   % BG
            trainingImgs{cnt_img-1,3} = temp_name{3};   % BG
        end
        test_file = importdata(['test',num2str(current_split-1),'.csv']);
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