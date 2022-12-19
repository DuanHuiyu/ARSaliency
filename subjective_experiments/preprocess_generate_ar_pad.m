% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% code to generate AR images (imread and imresize/pad to new size)
% --------------------------------------------------------------------------
clc

clear all

img_size = [2560 2560];    % padding
path_write = '..\SARD\raw_stimuli\AR_all_pad\';

% img_size = [900 1440];    % no padding
% path_write = '..\SARD\raw_stimuli\AR_all\';

% ----------------------- resize and save raw image -----------------------
path_read = '..\SARD\raw_stimuli\AR_webpage\';

% path_read = '..\SARD\raw_stimuli\AR_natural\';

% path_read = '..\SARD\raw_stimuli\AR_graphic\';

if ~exist(path_write)
    mkdir(path_write)
end

ext = {'*.jpeg','*.jpg','*.png'};
images = [];
for i = 1:length(ext)
    images = [images; dir([path_read ext{i}])];
end

for cnt = 1:length(images)
    cnt
    
    img_name = [path_read images(cnt).name];
    [img, ~, alpha] = imread(img_name);
    raw_size = size(img);

    up = round((img_size(1)-raw_size(1))/2);
    down = img_size(1)-raw_size(1)-up;
    left = round((img_size(2)-raw_size(2))/2);
    right = img_size(2)-raw_size(2)-left;
    zero_u = zeros(up, raw_size(2), 3);    % up zero
    zero_d = zeros(down, raw_size(2), 3);    % down zero
    zero_l = zeros(img_size(1), left, 3);    % left zero
    zero_r = zeros(img_size(1), right, 3);    % right zero
    img2 = cat(1, zero_u, img, zero_d);    % cat 0 up and down
    img3 = cat(2, zero_l, img2, zero_r);    % cat 0 left and right

    imwrite(img3, [path_write,images(cnt).name]);
end
