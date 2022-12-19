% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% wrap and crop raw pano(360) bg images to captured size
% --------------------------------------------------------------------------

clc
clear all

addpath('utils\equi2cubic')

path_bg = '..\SARD\raw_stimuli\BG_all\';
path_write = '..\SARD\raw_stimuli\BG_cubic\';
img_names = dir([path_bg '*.jpg']);

if exist(path_write)==0
	mkdir(path_write);
end

for cnt=1:length(img_names)
    cnt
    img_name=img_names(cnt).name;
    img = imread([path_bg, img_name]);
    
    base_name = strsplit(img_name,'.');
    write_name = [base_name{1}, '.png'];
    img_cubics = equi2cubic(img, 2560);
    imwrite(img_cubics{1}, [path_write, write_name]);
end

%% crop and write
path_read = '..\SARD\stimuli\BG_cubic\';
path_write = '..\SARD\stimuli\BG_crop\';

if exist(path_write)==0
	mkdir(path_write);
end

img_names = dir([path_read '*.png']);

for cnt=1:size(img_names,1)
    cnt
    img_name = img_names(cnt).name;
    img = imread([path_read,img_name]);
    r = centerCropWindow2d(size(img),[900,1440]);
    img_crop = imcrop(img,r);
    imwrite(img_crop, [path_write,img_name]);
end