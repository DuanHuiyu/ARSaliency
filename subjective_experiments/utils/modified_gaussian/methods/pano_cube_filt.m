function [ sal_map ] = pano_cube_filt( panorama,sigma )
%PANO_CUBE_FILT Takes an input equirectangular panoram (double, and likely a fixation map),
%converts it to a cube map, applies a gaussian filter to each face, and then converts it back to 
%equirectangular format and returns it.
    
    new_imgH = size(panorama,2)/4;                 % horizontal resolution = width
    new_imgShort = new_imgH;    % vertical resolution = height
    fov = pi /2;          % horizontal angle of view

    deg_per_pixel = rad2deg(fov)/new_imgH;
    
    VectX = [ 0 -pi -pi/2 0 pi/2  0];
    VectY = [ pi/2 0 0 0 0  -pi/2 ];

    for i = 1:length(VectX)
        x = VectX(i);         % range [-pi,   pi]
        y = VectY(i);         % range [-pi/2, pi/2]

        % generate the crop
        warped_image = imgLookAt(panorama, x, y, new_imgH, fov );
        %warped_image = warped_image/255;
        warped_image = warped_image((new_imgH-new_imgShort)/2+(1:new_imgShort),:,:);

        cubeRep{i} = warped_image;
    end
    
    %loop through each cube image and filter it by gaussian
    %max for debug, visualization
    %max_val =0;
    for i = 1:6
       curr_cube = cubeRep{i};
       curr_cube = imgaussfilt(curr_cube,sigma*(1/deg_per_pixel),'FilterSize',2*ceil(3*(sigma*1/deg_per_pixel))+1,'Padding',0);
       cubeRep{i} = curr_cube;
       %max_val = max(max_val,max(curr_cube(:)));
    end
       
    
%     for i = 1:6
%        curr_cube = cubeRep{i};
%        %curr_cube = imgaussfilt(curr_cube,sigma*(1/deg_per_pixel),'FilterSize',2*ceil(3*(sigma*1/deg_per_pixel))+1,'Padding',0);
%        cubeRep{i} = curr_cube/max_val;
%        %max_val = max(max_val,max(curr_cube(:)));
%     end
        
    sal_map = zeros(size(panorama));
    for i = 1:6
        [sphereImg_temp, validMap_temp] = im2Sphere(cubeRep{i}, fov, 4*new_imgH, 2*new_imgH, VectX(i), VectY(i));
        sal_map(validMap_temp) = sphereImg_temp(validMap_temp);
    end
    

end

