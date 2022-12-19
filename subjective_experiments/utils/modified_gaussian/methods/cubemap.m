function [ sal_map ] = cubemap( fix_map, sigma )
%CUBEMAP Generate a saliency map based on cubemap filtered
%by gaussian, converts the input fixation map into cube, computed saliency
%at specified number of different rotations to avoid the edge discontinuities and combines
%them. Outputs the map in equirectangular format
   %Inputs:
   %fix_map: Fixation map, in equirectangular format
   %sigma: Std dev for gaussian filtering, specify in visual degrees
   %scale: Generally full image Kent filtering takes a long time, this
   %Outputs: sal_map, saliency normalized between 0 and 1 in
   %equirectangular format.
   
   [image_rows,image_cols] = size(fix_map);
   
   %First filter an unrotated unequirectangular fixation map
   sal_map_1 = pano_cube_filt(fix_map,sigma);
   
   %Generate weights
   equi_weights_1 = gen_cube_weights(image_cols/4);
   
   %Rotate fixation map, then compute saliency
   xrot_cube = pi/4;
   yrot_cube = 0;
   zrot_cube = pi/4;
   R = eul2rotm([xrot_cube yrot_cube zrot_cube]);
   [rotated_fix_map,~] = rotatePanorama( fix_map, 0, R );
   
   sal_map_rot = pano_cube_filt(rotated_fix_map,sigma);
   
   %rotate saliency map back, generate weights rotated back
   sal_map_2 = rotatePanorama(sal_map_rot,0,inv(R));
   equi_weights_2 = rotatePanorama(equi_weights_1,0,inv(R));
   
   max_1 = max(sal_map_1(:));
   max_2 = max(sal_map_2(:));
    
   %
   sal_map_1 = sal_map_1/max_1;
   sal_map_2 = sal_map_2/max_2;
   
   %combine
   sal_map = (equi_weights_1.*sal_map_1 + equi_weights_2.*sal_map_2)./(equi_weights_1+equi_weights_2);
   
   %for debug
%    subplot(1,3,1);
%    imshow(sal_map_1);
%    title('Sal map 1')
%    subplot(1,3,2);
%    imshow(sal_map_2);
%    title('Sal map 2')
%    subplot(1,3,3);
%    imshow(sal_map);
%    title('Combined');
   
   
end

