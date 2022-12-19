% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% generate heat maps.
% --------------------------------------------------------------------------

clc
clear all

addpath('utils\signatureSal')   % contains utils for heatmap plot (heatmap_overlay.m)

img_path = '..\SARD\raw_stimuli\Superimposed2\';
img_names = dir([img_path '*.png']);
sal_path = '..\SARD\raw_subjective_data\process_results\fixMaps\';
write_path = '..\SARD\raw_subjective_data\process_results\heatMaps3_sal\';
write_path2 = '..\SARD\raw_subjective_data\process_results\heatMaps3_overlay\';

if exist(write_path)==0
	mkdir(write_path);
end
if exist(write_path2)==0
	mkdir(write_path2);
end

for cnt=1:length(img_names)
    cnt
    
    img_name=img_names(cnt).name;
    img = imread([img_path, img_name]);
    base_name = strsplit(img_name,'.');
    salimg_name = [base_name{1}, '_fixMap.png'];    % need to ensure the salmap name here
    sal = imread([sal_path, salimg_name]);
    [omap,cmap] = heatmap_overlay(img, sal, 'jet');
    
    save_name = [write_path, img_name];
    imwrite(cmap, save_name)
    save_name = [write_path2, img_name];
    imwrite(omap, save_name)
end
