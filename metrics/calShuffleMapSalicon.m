% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% make shuffle map
% --------------------------------------------------------------------------

clc
clear

pathDataset = '..\Salicon\val\fixpts_map\';
fixpts_list=dir([pathDataset, '*.png']);
ImgNum = length(fixpts_list);

mapSize = [1024 1024];
img_avg = zeros(mapSize);

for cnt = 1:ImgNum
    cnt
    img = im2double(imread([pathDataset, fixpts_list(cnt).name]));
    img = imresize(img, mapSize, 'nearest');
    img_avg = img_avg | img;
end

imwrite(img_avg, 'Salicon_shuffle_map.png');