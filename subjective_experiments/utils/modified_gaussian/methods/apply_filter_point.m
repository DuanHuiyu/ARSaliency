function [ ret_val ] = apply_filter_point( r,c, img, kernel)
%APPLY_FILTER_POINT Summary of this function goes here
%   Given a kernel and center point apply weighted sum of kernel with the
%   surrounding neighborhood of the pixels.

kernel_size = length(kernel);
sub_rows = -floor(kernel_size/2):floor(kernel_size/2);
sub_cols = -floor(kernel_size/2):floor(kernel_size/2);
sub_img = img(sub_rows+r,sub_cols+c);
sub_img = kernel .* sub_img;

ret_val = sum(sub_img(:));
end

