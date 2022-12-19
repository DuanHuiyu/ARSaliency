% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% -------------------------------------------------------------------------
% code to generate BG 360 images (copy and rename from raw folder)
% -------------------------------------------------------------------------

clc
clear all

img_size = [900 1440];
img_size2 = 500;

% ----------------------- resize and save raw image -----------------------
suffix = '';
path_read = '..\SARD\raw_imgs\raw_BG\';
path_write = '..\SARD\raw_stimuli\BG_all\';

if ~exist(path_write)
    mkdir(path_write)
end

ext = {'*.jpeg','*.jpg','*.png'};
images = [];
for i = 1:length(ext)
    images = [images; dir([path_read ext{i}])];
end

name_all=[];
for cnt = 1:length(images)
    cnt
    name = floor(mod(cnt-1,75)/25)*150 + (ceil(cnt/75)-1)*25 + (mod(cnt-1,25)+1);
    name_all = [name_all name];
    
    temp = split(images(cnt).name,'_');
    suffix = temp{1};
    if ((cnt>150) && (cnt <301))
        suffix = temp{2};
    end
    copyfile([path_read images(cnt).name],[path_write,'P',num2str(name),'_',suffix,'.jpg']);
end
