function [ kernel ] = kent_kernel( r,c,kernel_size,kappa,img_width_px,img_height_px )
%KENT_KERNEL Generates a kernel (non-symmetric)based on Kent pdf
    %Inputs:
    % r,c: row and column pixel of center from specified width,height
    % kernel_size is the size of a square kernel to generate, should be odd
    % 
    
    img_width_rad = deg2rad(360);
    img_height_rad = deg2rad(180);
    
    a=linspace(-floor(kernel_size/2),floor(kernel_size/2),kernel_size);
    b=linspace(-floor(kernel_size/2),floor(kernel_size/2),kernel_size);

    %get long, lat of center x,y
    long = img_width_rad * (c - img_width_px/2) / img_width_px;
    lat = img_height_rad * (r - img_height_px/2) / img_height_px;
    
    %match matlab's coordinate system
    long = long;
    lat = -lat;

    [c_x,c_y,c_z] = sph2cart(long,lat,1.0);
    center = [c_x c_y c_z];

    %center = center/norm(center);
    
    %adjust to indicies for current center
    a = a + r;
    b = b + c;
    
    %get long, lat of kernel points
    longs = img_width_rad * (b - img_width_px/2) / img_width_px;
    lats = img_height_rad * (a - img_height_px/2) / img_height_px;
    
    % match matlabs spherical coordinate scheme
    longs = longs;
    lats = -lats;
    
    [longs_mesh,lats_mesh] = meshgrid(longs,lats);

    [c_x,c_y,c_z] = sph2cart(longs_mesh(:),lats_mesh(:),1.0);
    xs = [c_x c_y c_z];

    kent = kent_pdf(xs,kappa,0,center);
    
    kent = kent/sum(kent(:));

    %need to convert from hpf to doubles if it was used
    kernel = double(kent);
    kernel = reshape(kernel,[kernel_size kernel_size]);
end

