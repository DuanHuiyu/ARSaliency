%add folders to path
addpath(genpath('methods'));
addpath(genpath('PanoBasic'));

%load a test fixation map, and scanpaths for viewport method
load('demo_data.mat');

%isotropic gaussian. Third parameter controls use of modified or naive gaussian
sal_map_g = gauss(fix_map,sigma,false);

%Modified Gaussian. We down sample the image first, and then apply.
sal_map_modified_g = gauss(imresize(fix_map,scale),sigma,true);
sal_map_modified_g = imresize(sal_map_modified_g,[image_rows image_cols]);

%cubemap method
sal_map_cube = cubemap( fix_map, sigma );

%viewport method
%This will take a very long time, and is not reccomended!!!
sal_map_vp = viewport(N,hiw_fix,sigma_viewport,image_cols,image_rows, screen_fix );

%Kent based method. Run at 5% scale for this call.
sal_map_k = kent(fix_map,kappa,scale);
