function [ sal_map_resized ] = kent( fix_map,kappa,scale )
%Kent Generate a saliency map based on Kent filtering of
%equirectangular map.
   %Inputs:
   %fix_map: Fixation map
   %kappa: Param for Kent equation, analgolous to sigma for gaussian
   %scale: Generally full image Kent filtering takes a long time, this is
   %specificed scale in (0,1] to perform at reduced scale (.5 = half size)
   %row approximation is valid and less than machine eps
   %Outputs: sal_map, saliency normalized between 0 and 1
   if nargin <3 
     scale = .05; 
   end
 

   [image_rows,image_cols] = size(fix_map);
   %NOTE: degree per pixel is computed at full resolution before shrinking.
   %This is used to define only the kernel width, which must be large to
   %capture kernels that are stretched far near the top and bottom.
   deg_per_pixel = 360/image_cols;
   
   %Kernel size needs to be big enough for all of kernel values.
   %for now it is twenty times 7 degrees visual angle at full resolution.
   kernel_size = ceil(20*7.0*(1/deg_per_pixel));
   kernel_size = kernel_size * scale;
   kernel_size = 2*floor(kernel_size/2)+1;
   
   %scale map
   fix_map_r = imresize(fix_map,scale);
   [image_rows_scaled, image_cols_scaled] = size(fix_map_r);
   saliency_map = zeros(image_rows_scaled,image_cols_scaled);
   
   
   %pad map with with circular values, to apply filters along boundaries
   %This padding works for the naive apply_filter point method 
   fix_data_r_pad = padarray(fix_map_r,[ceil(kernel_size/2) ceil(kernel_size/2)],'circular');

   %Use this pad array for the FFT version, similar to gauss.m. This works at reduced
   %scales but can cause out of memory errors for full size 360 imgs
   %fix_data_r_pad = padarray(fix_map_r,[ceil(kernel_size/2) 0],'circular');

   %parfor r= 1:image_rows_scaled
   for r= 1:image_rows_scaled
     kernel = kent_kernel(r,floor(image_cols_scaled/2),kernel_size,kappa,image_cols_scaled,image_rows_scaled);

     %This block computes a convolution via fft. 
     %It performs faster than looping over the row, but may use more
     %memory
     %NOTE: to use this method need to use the following call to padarray
     %fix_data_pad = padarray(fix_map,[ceil(kernel_size/2) ceil(kernel_size/2)],'circular');
     for  c = 1:image_cols_scaled
        %offsets added to r,c for padding
        saliency_val = apply_filter_point(r+ceil(kernel_size/2),c+ceil(kernel_size/2),fix_data_r_pad,kernel);
        saliency_map(r,c) = saliency_val;
     end 

   %This block computes a convolution via fft. 
   %It performs faster than looping over the row, but may use more
   %memory
   %NOTE: to use this method need to use the following call to padarray
   %fix_data_pad = padarray(fix_map,[ceil(kernel_size/2) 0],'circular');
%    offset = (kernel_size-1)/2 ;
% 
%    sub_img = fix_data_r_pad(r+ ceil(kernel_size/2)-offset:r+ceil(kernel_size/2)+offset,:);
%    sub_sal = conv_fft2(sub_img,rot90(kernel,2),'wrap');
%    row_val = sub_sal(offset+1,:);
%    saliency_map(r,:) = row_val;

   end

   sal_map_resized = imresize(saliency_map,size(fix_map));
   sal_map_resized = sal_map_resized/(sum(sal_map_resized(:)));
   
end

