% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% generate fixation data from raw eye gaze data
% -------------------------------------------------------------------------

clc
clear all

addpath('utils')

path_pano_img = '..\SARD\raw_stimuli\BG_all\';
path_mixed_img = '..\SARD\raw_subjective_data\captured\';
path_eyedata = '..\SARD\raw_subjective_data\gaze\';

path_write = '..\SARD\raw_subjective_data\process_results\fixations\';
if ~exist(path_write)
    mkdir(path_write)
end

img_names = dir([path_mixed_img '*.png']);

for cnt=1:size(img_names,1)
    cnt
    % read current image
    img_name = img_names(cnt).name;
    % img_mixed = imread([path_mixed_img,img_name]);
    % read current file
    split_name = split(img_name,'.');
    files = dir(fullfile(path_eyedata,[split_name{1},'*.csv']));
    %% draw head eye map
    for cnt2 = 1:size(files,1)
        data = xlsread([path_eyedata,files(cnt2).name]);

        x_eye = data(:,2);
        y_eye = data(:,3);
        z_eye = data(:,4);
        x_head = data(:,5);
        y_head = data(:,6);
        z_head = data(:,7);

        for cnt3 = 1:size(data,1)
            [headLat(cnt3,cnt2), headLong(cnt3,cnt2)] = headCoordinateMap(x_head(cnt3), y_head(cnt3), z_head(cnt3));
            [eyeLat(cnt3,cnt2), eyeLong(cnt3,cnt2)] = eyeCoordinateMap(x_eye(cnt3), y_eye(cnt3), z_eye(cnt3));
        end
        % --- fixation cluster ---
        [fixation_position,fixation_times,fixation_duration] = calPanoGaze2Fixation(eyeLat(:,cnt2)',eyeLong(:,cnt2)');
        % --- write fixation information ---
        fixation_info = [fixation_position',fixation_times',fixation_duration'];
        csvwrite([path_write,split_name{1},'_subject',num2str(cnt2),'.csv'],fixation_info);
    end
end