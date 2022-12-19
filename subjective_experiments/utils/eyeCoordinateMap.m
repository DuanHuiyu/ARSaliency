function [lat, long] = eyeCoordinateMap(x, y, z)
% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
    lat_shift = 90.0;
    
    length_vector = sqrt((x^2+y^2+z^2));
    lat = lat_shift - asin(y/length_vector)/pi*180;
    
    atan_theta = atan(z/x)/pi*180;
    tmp_theta = (x >= 0) * 90.0 + (x < 0) * 270.0 - atan_theta;
    %long = ((tmp_theta + 90.0) <= 360.0) * (tmp_theta + 90.0) ...
    %            + ((tmp_theta + 90.0) > 360.0) * (tmp_theta + 90.0 - 360.0);
    long = mod(tmp_theta+90,360);
    
    
end