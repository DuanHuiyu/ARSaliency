function [ sal_map ] = gauss( fix_map,sigma,is_distorted_gauss )
%GAUSS Generate a saliency map based on Gaussian filtering of
%equirectangular map. Note for the associated AIVR paper we sub-sampled our
%fixation maps prior to calling this function.
   %Inputs:
   %fix_map: Fixation map
   %sigma: std dev value in degrees to use for Gaussian kernel
   %is_distorted_gauss: boolean for if a scaled gaussian to match equirectangualr
   %distortions should be used
   %Outputs: sal_map, saliency normalized between 0 and 1
   if nargin <3 
     is_distorted_gauss = true; 
   end
    
   [image_rows,image_cols] = size(fix_map);
   deg_per_pixel = 360/image_cols;
   
   if (~is_distorted_gauss)
        sal_map = imgaussfilt(fix_map,sigma*(1/(deg_per_pixel)),'Padding','circular','FilterSize',2*ceil(3*sigma*(1/(deg_per_pixel)))+1);
        
        sal_map = sal_map/max(sal_map(:));
   else
       %Kernels are scaled larger than needed to capture stretched
       %gaussians near the top and bottom of image.
       kernel_size = ceil(12*7.0*(1/deg_per_pixel));
	   kernel_size = 2*floor(kernel_size/2)+1;
       gauss_k = fspecial('gaussian',kernel_size,sigma*(1/deg_per_pixel));
       sal_map = zeros(image_rows,image_cols);
       fix_data_pad = padarray(fix_map,[ceil(kernel_size/2) ceil(kernel_size/2)],'circular');
       %fix_data_pad = padarray(fix_map,[ceil(kernel_size/2) 0],'circular');
   
       %parfor r = 1:image_rows
       for r = 1:image_rows
           %compute scaled gaussian
           elev_angle = deg2rad(180)*abs(r/image_rows - 0.5);

           scale_x = 1/cos(elev_angle);
           sigma_x = scale_x*sigma;
           sigma_y = sigma;
           h = fspecial( 'gaussian', [1 kernel_size], sigma_x*(1/deg_per_pixel) ); % horizontal
           v = fspecial( 'gaussian', [kernel_size 1], sigma_y*(1/deg_per_pixel) ); % vertical
           gauss_k = v*h;
           
           %This block computes a convolution via fft. 
           %It performs faster than looping over the row, but may use more
           %memory
           %NOTE: to use this method need to use the following call to padarray above
           fix_data_pad = padarray(fix_map,[ceil(kernel_size/2) 0],'circular');
           offset = round((kernel_size-1)/2) ;
           sub_img = fix_data_pad(r+ ceil(kernel_size/2)-offset:r+ceil(kernel_size/2)+offset,:);
           sub_sal = conv_fft2(sub_img,gauss_k,'reflect');
           row_val = sub_sal(offset+1,:);
           sal_map(r,:) = row_val;
           
%           %This block computes a convolution via a loop over the row. 
            %NOTE: to use this method need to use the following call to padarray above
            %fix_data_pad = padarray(fix_map,[ceil(kernel_size/2) ceil(kernel_size/2)],'circular');
%             for c = 1:image_cols
%                 %offsets added to r,c for padding
%                 saliency_val = apply_filter_point(r+ceil(kernel_size/2),c+ceil(kernel_size/2),fix_data_pad,gauss_k);
%                 sal_map(r,c) = saliency_val;
%             end
       end
       
       sal_map = sal_map/max(sal_map(:));   % huiyu changed below line for better visualization
       % sal_map = sal_map/sum(sal_map(:));
   end
end

