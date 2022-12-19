% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% generate eye fixation map, fixation density map, heat map from fixation
% data.
% --------------------------------------------------------------------------

clc
clear all

addpath('utils\modified_gaussian\methods')
addpath('utils\conv_fft2')
addpath('utils\signatureSal')   % contains utils for heatmap plot (heatmap_overlay.m)

path_pano_img = '..\SARD\raw_stimuli\BG_all\';
path_mixed_img = '..\SARD\raw_subjective_data\captured\';
path_eyedata = '..\SARD\raw_subjective_data\process_results\fixations\';

path_write_fixPts = '..\SARD\raw_subjective_data\process_results\fixPts\';
path_write_fixMaps = '..\SARD\raw_subjective_data\process_results\fixMaps\';
path_write_heatMaps = '..\SARD\raw_subjective_data\process_results\heatMaps1\';
path_write_heatMaps2 = '..\SARD\raw_subjective_data\process_results\heatMaps2\';

crop = 1;   % crop fixations
if crop
    path_write_fixPts = '..\SARD\raw_subjective_data\process_results\fixPts_crop\';
    path_write_fixMaps = '..\SARD\raw_subjective_data\process_results\fixMaps_crop\';
    path_write_heatMaps = '..\SARD\raw_subjective_data\process_results\heatMaps1_crop\';
    path_write_heatMaps2 = '..\SARD\raw_subjective_data\process_results\heatMaps2_crop\';
end

if exist(path_write_fixPts)==0
	mkdir(path_write_fixPts);
end
if exist(path_write_fixMaps)==0
	mkdir(path_write_fixMaps);
end
if exist(path_write_heatMaps)==0
	mkdir(path_write_heatMaps);
end
if exist(path_write_heatMaps2)==0
	mkdir(path_write_heatMaps2);
end

img_names = dir([path_mixed_img '*.png']);

for cnt=1:size(img_names,1)
    cnt
    img_name = img_names(cnt).name;
    img_mixed = imread([path_mixed_img,img_name]);
    split_name = split(img_name,'_');
    img_pano = imread([path_pano_img,split_name{1},'_',split_name{2},'.jpg']);
    img_pano = imresize(img_pano,[512 1024]); % resize to 512*1024
    [imgRow,imgCol,~] = size(img_pano);
    [imgMixedRow,imgMixedCol,~] = size(img_mixed);
    
    % read current file
    split_name = split(img_name,'.');
    files = dir(fullfile(path_eyedata,[split_name{1},'*.csv']));
    
    %% calculate latitude and longtitude of eye movements
    eyeLat = [];
    eyeLong = [];
    cnt23 = 0;
    for cnt2 = 1:size(files,1)
        data = xlsread([path_eyedata,files(cnt2).name]);
        
        for cnt3 = 1:size(data,1)
            cnt23 = cnt23 + 1;
            eyeLat(cnt23,1) = data(cnt3,1);
            eyeLong(cnt23,1) = data(cnt3,2);
        end
    end
    
    %% draw head eye map for captured image
    % make sure eye data is limited to range eyeLat[45,135] & eyeLong[135,225],
    % since we mainly concern the perceptual viewport 
    % rather than the whole pano image.
    temp_eyeLat = eyeLat;
    temp_eyeLong = eyeLong;
    eyeLat([find(temp_eyeLat<=45);find(temp_eyeLat>=135);find(temp_eyeLong<=135);find(temp_eyeLong>=225)]) = [];
    eyeLong([find(temp_eyeLat<=45);find(temp_eyeLat>=135);find(temp_eyeLong<=135);find(temp_eyeLong>=225)]) = [];
    
    % from equirectangular to front-cubic
    eyeXImg = round(imgMixedCol * (tan((eyeLong-180)/180*pi) + 1) / 2);
    eyeYImg = round(imgMixedRow * (tan((eyeLat-90)/180*pi)./cos((eyeLong-180)/180*pi) + 1) / 2);
    % make sure data include in eyeXImg[0,imgMixedCol] & eyeYImg[0,imgMixedRow]
    temp_eyeXImg = eyeXImg;
    temp_eyeYImg = eyeYImg;
    eyeXImg([find(temp_eyeXImg<=0);find(temp_eyeXImg>=imgMixedCol);find(temp_eyeYImg<=0);find(temp_eyeYImg>=imgMixedRow)]) = [];
    eyeYImg([find(temp_eyeXImg<=0);find(temp_eyeXImg>=imgMixedCol);find(temp_eyeYImg<=0);find(temp_eyeYImg>=imgMixedRow)]) = [];

    eyeFixMap = calFixMap(imgMixedRow, imgMixedCol, eyeYImg, eyeXImg);
    if crop
        r = centerCropWindow2d(size(eyeFixMap),[900,1440]);
        eyeFixMap = imcrop(eyeFixMap,r);
        img_mixed = imcrop(img_mixed,r);
    end
    % figure(4)
    % imshow(eyeFixMap)
    imwrite(eyeFixMap,[path_write_fixPts,split_name{1},'_fixPts.png']);
    
    eyeSalMap = calSalMap(eyeFixMap, imgMixedCol, 90);
    % figure(5)
    % imshow(eyeSalMap)
    imwrite(eyeSalMap,[path_write_fixMaps,split_name{1},'_fixMap.png']);
    
    eyeHeatMap = calPanoHeatMap(img_mixed, eyeSalMap);
    saveas(eyeHeatMap,[path_write_heatMaps,split_name{1},'_heatMap.png']);
    
    heat = heatmap_overlay(img_mixed, eyeSalMap, 'jet');
    imwrite(heat, [path_write_heatMaps2,split_name{1},'_heatMap.png']);
    
    close all
end

%% for small size fixation writing
% path_mixed_img = 'E:\research\ARsaliency\ar_saliency_database\small_size\Superimposed\';
% path_eyedata = 'E:\research\ARsaliency\Data\process_results\fixations\';
% path_write_fixPts = 'E:\research\ARsaliency\Data\process_results\fixPts_smallsize\';
% crop = 1;   % crop fixations
% if crop
%     path_write_fixPts_crop = 'E:\research\ARsaliency\Data\process_results\fixPts_crop_smallsize\';
% end
% if exist(path_write_fixPts)==0
% 	mkdir(path_write_fixPts);
% end
% if exist(path_write_fixPts_crop)==0
% 	mkdir(path_write_fixPts_crop);
% end
% img_names = dir([path_mixed_img '*.png']);
% 
% for cnt=1:size(img_names,1)
%     cnt
%     img_name = img_names(cnt).name;
%     img_mixed = imread([path_mixed_img,img_name]);
%     [imgMixedRow,imgMixedCol,~] = size(img_mixed);
%     
%     % read current file
%     split_name = split(img_name,'.');
%     files = dir(fullfile(path_eyedata,[split_name{1},'*.csv']));
%     
%     %% calculate latitude and longtitude of eye movements
%     eyeLat = [];
%     eyeLong = [];
%     cnt23 = 0;
%     for cnt2 = 1:size(files,1)
%         data = xlsread([path_eyedata,files(cnt2).name]);
%         
% %         eyeLat(:,cnt2) = [eyeLat,data(:,1)];
% %         eyeLong(:,cnt2) = [eyeLong,data(:,1)];
%         for cnt3 = 1:size(data,1)
%             cnt23 = cnt23 + 1;
%             eyeLat(cnt23,1) = data(cnt3,1);
%             eyeLong(cnt23,1) = data(cnt3,2);
%         end
%     end
%     
%     %% draw head eye map for captured image
%     % make sure data include in eyeLat[45,135] & eyeLong[135,225]
%     temp_eyeLat = eyeLat;
%     temp_eyeLong = eyeLong;
%     eyeLat([find(temp_eyeLat<=45);find(temp_eyeLat>=135);find(temp_eyeLong<=135);find(temp_eyeLong>=225)]) = [];
%     eyeLong([find(temp_eyeLat<=45);find(temp_eyeLat>=135);find(temp_eyeLong<=135);find(temp_eyeLong>=225)]) = [];
%     
%     % from equirectangular to front-cubic
%     eyeXImg = round(imgMixedCol * (tan((eyeLong-180)/180*pi) + 1) / 2);
%     eyeYImg = round(imgMixedRow * (tan((eyeLat-90)/180*pi)./cos((eyeLong-180)/180*pi) + 1) / 2);
%     % make sure data include in eyeXImg[0,imgMixedCol] & eyeYImg[0,imgMixedRow]
%     temp_eyeXImg = eyeXImg;
%     temp_eyeYImg = eyeYImg;
%     eyeXImg([find(temp_eyeXImg<=0);find(temp_eyeXImg>=imgMixedCol);find(temp_eyeYImg<=0);find(temp_eyeYImg>=imgMixedRow)]) = [];
%     eyeYImg([find(temp_eyeXImg<=0);find(temp_eyeXImg>=imgMixedCol);find(temp_eyeYImg<=0);find(temp_eyeYImg>=imgMixedRow)]) = [];
% 
%     eyeFixMap = calFixMap(imgMixedRow, imgMixedCol, eyeYImg, eyeXImg);
%     if crop
%         r = centerCropWindow2d(size(eyeFixMap),[180,288]);
%         eyeFixMap_crop = imcrop(eyeFixMap,r);
%     end
%     % figure(4)
%     % imshow(eyeFixMap)
%     imwrite(eyeFixMap,[path_write_fixPts,split_name{1},'.png']);
%     imwrite(eyeFixMap_crop,[path_write_fixPts_crop,split_name{1},'.png']);
% end