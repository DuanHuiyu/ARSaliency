% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% code to generate AR images (imread and imresize/pad to new size)
% --------------------------------------------------------------------------

clc
clear all

img_size = [900 1440];  % resize all images to [H, W] [900 1440]
img_size2 = 500;    % for graphic images, we resize them to 500 and pad 
                    % with transparent images

% ----------------------- resize and save raw image -----------------------
mode = 0;
suffix = '_webpage';
path_read = '..\SARD\raw_imgs\raw_webpage\';
path_write = '..\SARD\raw_stimuli\AR_webpage\';

% mode = 0;
% suffix = '_natural';
% path_read = '..\SARD\raw_imgs\raw_natural\';
% path_write = '..\SARD\raw_stimuli\AR_natural\';

% mode = 1;
% suffix = '_graphic';
% path_read = '..\SARD\raw_imgs\raw_graphic\';
% path_write = '..\SARD\raw_stimuli\AR_graphic\';

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
    
    images(cnt).name = [path_read images(cnt).name];
    if mode == 1    % save transparent .png
        [img, ~, alpha] = imread(images(cnt).name);
        temp_size = size(size(img));
        if temp_size(2) == 2
            img = cat(3, img, img, img);
        end
        raw_size = size(img);
        
        if max(raw_size)>img_size2
            new_size = [raw_size(1)/max(raw_size) raw_size(2)/max(raw_size)];
            new_size = ceil(new_size * img_size2);
            img2 = imresize(img, new_size);
            alpha2 = imresize(alpha, new_size);
        else
            new_size = raw_size;
            img2 = img;
            alpha2 = alpha;
        end
        
        up = randi(img_size(1)-new_size(1));
        down = img_size(1)-new_size(1)-up;
        left = randi(img_size(2)-new_size(2));
        right = img_size(2)-new_size(2)-left;
        zero_u = zeros(up, new_size(2), 3);    % up zero
        zero_d = zeros(down, new_size(2), 3);    % down zero
        zero_l = zeros(img_size(1), left, 3);    % left zero
        zero_r = zeros(img_size(1), right, 3);    % right zero
        img3 = cat(1, zero_u, img2, zero_d);    % cat 0 up and down
        img4 = cat(2, zero_l, img3, zero_r);    % cat 0 left and right
        
        zero_u = zeros(up, new_size(2), 1);    % up zero
        zero_d = zeros(down, new_size(2), 1);    % down zero
        zero_l = zeros(img_size(1), left, 1);    % left zero
        zero_r = zeros(img_size(1), right, 1);    % right zero
        alpha3 = cat(1, zero_u, alpha2, zero_d);    % cat 0 up and down
        alpha4 = cat(2, zero_l, alpha3, zero_r);    % cat 0 left and right
        
        imwrite(img4, [path_write,'P',num2str(cnt),suffix,'.png'], 'Alpha', alpha4);
    else
        img = im2double(imread(images(cnt).name))*255;
        img = imresize(img,img_size);
        imwrite(img/255, [path_write,'P',num2str(cnt),suffix,'.png']);
    end
end
