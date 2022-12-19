% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------

clc
clear all

path = '..\SARD\large_size\fixMaps\';
% path = '..\SARD\large_size\fixMaps_crop\';
% path = '..\SARD\large_size\fixPts\';
% path = '..\SARD\large_size\fixPts_crop\';
% path = '..\SARD\small_size\fixMaps\';
% path = '..\SARD\small_size\fixMaps_crop\';

names = dir([path '*.png']);
for cnt=1:size(names,1)
    cnt
    name = names(cnt).name;
    new_name = erase(name,'_fixMap');
    new_name = erase(new_name,'_fixPts');
    movefile([path,name], [path,new_name]);
end