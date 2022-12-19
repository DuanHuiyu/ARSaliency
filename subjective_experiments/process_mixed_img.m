% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% code to process captured mixed images, including "crop" and "save"
% --------------------------------------------------------------------------

clc
clear all

path_mixed_img = '..\SARD\raw_subjective_data\raw_captured\';
path_mixed_write = '..\SARD\raw_subjective_data\cropped\';
path_mixed_write2 = '..\SARD\raw_subjective_data\captured\';
if exist(path_mixed_write)==0
	mkdir(path_mixed_write);
end
if exist(path_mixed_write2)==0
	mkdir(path_mixed_write2);
end

img_names = dir([path_mixed_img '*.png']);

for cnt=1:size(img_names,1)
    cnt
    img_name = img_names(cnt).name;
    img = imread([path_mixed_img,img_name]);
    split_name = split(img_name,'__');
    r = centerCropWindow2d(size(img),[900,1440]);
    img_crop = imcrop(img,r);
    imwrite(img_crop, [path_mixed_write,split_name{2}]);
    imwrite(img, [path_mixed_write2,split_name{2}]);
end

%% check
path_pano_img = '..\SARD\raw_stimuli\BG_all\';
path_ar_img1 = '..\SARD\raw_stimuli\AR_graphic\';
path_ar_img2 = '..\SARD\raw_stimuli\AR_natural\';
path_ar_img3 = '..\SARD\raw_stimuli\AR_webpage\';
for cnt = 1:450
    cnt
    img_pano = dir([path_pano_img 'P' num2str(cnt) '_*.jpg']);
    if cnt<151
        img_ar = dir([path_ar_img1 'P' num2str(cnt) '_*.png']);
    end
    if cnt>=151 && cnt<301
        img_ar = dir([path_ar_img2 'P' num2str(cnt-150) '_*.png']);
    end
    if cnt>=301
        img_ar = dir([path_ar_img3 'P' num2str(cnt-300) '_*.png']);
    end
    
    img_pano_name = img_pano.name;
    img_ar_name = img_ar.name;
    img_pano_name_split = split(img_pano_name,'.');
    img_ar_name_split = split(img_ar_name,'.');
    img_crop1 = imread([path_mixed_write2,img_ar_name_split{1},'_',img_pano_name_split{1},'_0.png']);
    img_crop1 = imread([path_mixed_write2,img_ar_name_split{1},'_',img_pano_name_split{1},'_1.png']);
    img_crop1 = imread([path_mixed_write2,img_ar_name_split{1},'_',img_pano_name_split{1},'_2.png']);
    img_mixed1 = imread([path_mixed_write2,img_ar_name_split{1},'_',img_pano_name_split{1},'_0.png']);
    img_mixed2 = imread([path_mixed_write2,img_ar_name_split{1},'_',img_pano_name_split{1},'_1.png']);
    img_mixed3 = imread([path_mixed_write2,img_ar_name_split{1},'_',img_pano_name_split{1},'_2.png']);
end