% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% resize all image for deep learning method
% --------------------------------------------------------------------------
clc
clear all

path_read = {'..\SARD\large_size\AR\'
    '..\SARD\large_size\AR_crop\'
    '..\SARD\large_size\BG\'
    '..\SARD\large_size\BG_crop\'
    '..\SARD\large_size\Superimposed\'
    '..\SARD\large_size\Superimposed_crop\'
    '..\SARD\large_size\fixMaps\'
    '..\SARD\large_size\fixMaps_crop\'};

path_write = {'..\SARD\small_size\AR\'
    '..\SARD\small_size\AR_crop\'
    '..\SARD\small_size\BG\'
    '..\SARD\small_size\BG_crop\'
    '..\SARD\small_size\Superimposed\'
    '..\SARD\small_size\Superimposed_crop\'
    '..\SARD\small_size\fixMaps\'
    '..\SARD\small_size\fixMaps_crop\'};

scale = 5;

for cnt = 1:length(path_read)
    cnt
    if ~exist(path_write{cnt})
        mkdir(path_write{cnt})
    end

    ext = {'*.jpeg','*.jpg','*.png'};
    images = [];
    for i = 1:length(ext)
        images = [images; dir([path_read{cnt} ext{i}])];
    end
    
    for cnt2 = 1:length(images)
        img_name = [path_read{cnt},images(cnt2).name];
        [img, ~, alpha] = imread(img_name);
        raw_size = size(img);
        
        new_size = [raw_size(1)/scale raw_size(2)/scale];
        img_resized = imresize(img, new_size);
        
        imwrite(img_resized, [path_write{cnt},images(cnt2).name]);
    end
end