% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% rewrite names for better viewing
% raw: ar_bg_alpha.png / ar_bg_alpha_subjectname.csv
% after: bg_ar_alpha.png / bg_ar_alpha_subjectname.csv
% --------------------------------------------------------------------------

%%
clc
clear all

path_eyedata = '..\SARD\raw_subjective_data\gaze\';
path_mixedimg1 = '..\SARD\raw_subjective_data\captured\';
path_mixedimg2 = '..\SARD\raw_subjective_data\cropped\';

%% rename eye tracking data files
names = dir([path_eyedata '*.csv']);
for cnt=1:size(names,1)
    cnt
    name = names(cnt).name;
    split_name = split(name,'_');
    new_name = [split_name{3},'_',split_name{4},'_',split_name{1},'_',split_name{2}];
    for cnt2 = 5:size(split_name,1)
        new_name = [new_name,'_',split_name{cnt2}];
    end
    movefile([path_eyedata,name], [path_eyedata,new_name]);
end

%% rename captured images
% names = dir([path_mixedimg1 '*.png']);
% for cnt=1:size(names,1)
%     cnt
%     name = names(cnt).name;
%     split_name = split(name,'_');
%     new_name = [split_name{3},'_',split_name{4},'_',split_name{1},'_',split_name{2}];
%     for cnt2 = 5:size(split_name,1)
%         new_name = [new_name,'_',split_name{cnt2}];
%     end
%     movefile([path_mixedimg1,name], [path_mixedimg1,new_name]);
% end

%% rename cropped images
% names = dir([path_mixedimg2 '*.png']);
% for cnt=1:size(names,1)
%     cnt
%     name = names(cnt).name;
%     split_name = split(name,'_');
%     new_name = [split_name{3},'_',split_name{4},'_',split_name{1},'_',split_name{2}];
%     for cnt2 = 5:size(split_name,1)
%         new_name = [new_name,'_',split_name{cnt2}];
%     end
%     movefile([path_mixedimg2,name], [path_mixedimg2,new_name]);
% end

%% check if images are complete
% path_pano_img = '..\SARD\raw_stimuli\BG_all\';
% path_ar_img1 = '..\SARD\raw_stimuli\AR_graphic\';
% path_ar_img2 = '..\SARD\raw_stimuli\AR_natural\';
% path_ar_img3 = '..\SARD\raw_stimuli\AR_webpage\';
% for cnt = 1:450
%     cnt
%     img_pano = dir([path_pano_img 'P' num2str(cnt) '_*.jpg']);
%     if cnt<151
%         img_ar = dir([path_ar_img1 'P' num2str(cnt) '_*.png']);
%     end
%     if cnt>=151 && cnt<301
%         img_ar = dir([path_ar_img2 'P' num2str(cnt-150) '_*.png']);
%     end
%     if cnt>=301
%         img_ar = dir([path_ar_img3 'P' num2str(cnt-300) '_*.png']);
%     end
%     img_pano_name = img_pano.name;
%     img_ar_name = img_ar.name;
%     img_pano_name_split = split(img_pano_name,'.');
%     img_ar_name_split = split(img_ar_name,'.');
%     img_captured = imread([path_mixedimg1,img_pano_name_split{1},'_',img_ar_name_split{1},'_0.png']);
%     img_captured = imread([path_mixedimg1,img_pano_name_split{1},'_',img_ar_name_split{1},'_1.png']);
%     img_captured = imread([path_mixedimg1,img_pano_name_split{1},'_',img_ar_name_split{1},'_2.png']);
%     img_cropped = imread([path_mixedimg2,img_pano_name_split{1},'_',img_ar_name_split{1},'_0.png']);
%     img_cropped = imread([path_mixedimg2,img_pano_name_split{1},'_',img_ar_name_split{1},'_1.png']);
%     img_cropped = imread([path_mixedimg2,img_pano_name_split{1},'_',img_ar_name_split{1},'_2.png']);
% end

%% check if eye tracking data are complete
names = dir([path_mixedimg2 '*.png']);
size(names,1)
cnt2 = 0;
for cnt=1:size(names,1)
    name = names(cnt).name;
    split_name = split(name,'.');
    image_name = split_name{1};
    gaze_name = dir([path_eyedata,image_name,'*.csv']);
    if size(gaze_name,1) ~= 20
        cnt
        image_name
        size(gaze_name,1)
        cnt2 = cnt2+1;
        image_missed_name{cnt2,1} = image_name;
        image_missed_cnt{cnt2,1} = size(gaze_name,1);
    end
end