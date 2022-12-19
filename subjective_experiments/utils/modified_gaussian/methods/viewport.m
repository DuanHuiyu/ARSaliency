function [ sal_map ] = viewport(N, hiw_fix,sigma,img_width,img_height, screen_fix )
%viewport Generates saliency map using viewport. Ensure that BasicPano code
%is included, as this needs functions from it.
%   Inputs:
%   hiw_fix: List of texture coordinates [x,y] normalized for head
%   location during fixation. Used to determine direction viewport is
%   facing for projection onto sphere. Each row is each fixation
%   sigma: Std. dev for gaussian when convolving on viewport, in visual
%   degrees
%   img_width/height: width and height of equirectangular image to generate
%   saliency map for.
%   screen_fix: List of gaze data in screen coordinates (x,y) for a 1920 x
%   1080 viewport. Values in this function must be changed for different
%   viewport sizes
%   Each row is a fixation. If not given then head based
%   saliency is used.

    %ensure screen and hiw_fix have same # of rows

    
    img_width_rad = deg2rad(360);
    img_height_rad = deg2rad(180);
    [num_fixations,~] = size(hiw_fix);
    viewport_width = 1920;
    viewport_height = 1080;
    vertical_fov = 106;
    horizontal_fov = 95;
    
    vert_deg_per_pixel = vertical_fov/1080;
    horiz_deg_per_pixel = horizontal_fov/1920;
    
    %If gaze param is omitted, use head based saliency
    %where we assume gaze in the center of the viewport
    if nargin <6
        %set screen fixation location to center for all fixations
        screen_center = [round(viewport_width/2) round(viewport_height/2)];
        screen_fix = repmat(screen_center,num_fixations,1);
    
    end
    
    %adjust normalized texture coordinates into lat/long
    hiw_longs = (hiw_fix(:,1)-0.5)*img_width_rad;
    hiw_lats = (hiw_fix(:,2)-0.5)*img_height_rad;
    
    sal_map = zeros(img_height,img_width);
    %Loop through each fixation, computing viewport saliency and projecting
    %onto sal_map
    screen_xs = screen_fix(:,1);
    screen_ys = screen_fix(:,2);
    for i = 1:num_fixations
 
        fix_x = round(screen_xs(i));
        fix_y = round(screen_ys(i));
        
        if (fix_x <= 0 || fix_x > img_width)
            continue;
        elseif (fix_y <= 0 || fix_y > img_height)
            continue;
        end
        
        hiw_tex_long = hiw_longs(i);
        hiw_tex_lat = hiw_lats(i);
        
        viewport = zeros(1080,1920);
        viewport(fix_y,fix_x) = 1;
        
        kernel_size = 2*ceil(3*sigma*(1/horiz_deg_per_pixel))+1;
        h = fspecial( 'gaussian', [1 kernel_size], sigma*(1/horiz_deg_per_pixel) ); % horizontal
        v = fspecial( 'gaussian', [kernel_size 1], sigma*(1/vert_deg_per_pixel) ); % vertical
        gauss_k = v*h;
        
        viewport = imfilter(viewport,gauss_k);
        viewport = viewport/(max(viewport(:)));

        [sphericalImg,ValidMap] = im2Sphere(viewport,deg2rad(horizontal_fov), img_width,img_height,hiw_tex_long,-hiw_tex_lat);
        
        sal_map(ValidMap) = sal_map(ValidMap) + sphericalImg(ValidMap)/N;

    end
    
    sal_map = sal_map/(max(sal_map(:)));
    
    %Due to many small values the viewport is best viewed by normalizing
    %between 0 and 1, and not summing to 1. For comparison the normalization 
    %by summing to one was used.
    %sal_map = sal_map/(sum(sal_map(:)));
    
    
end

