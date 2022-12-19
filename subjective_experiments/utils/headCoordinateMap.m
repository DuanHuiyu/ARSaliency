function [lat, long] = headCoordinateMap(x, y, z)
% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
    if (x>=90)&&(x<270) %x
        %long = y-180; %y %latitude
        long = mod(y+270,360); %y %latitude
        lat = 270-x; %x %longitude
    elseif (x>=270)&&(x<=360)
        %long = y;
        long = mod(y+90,360);
        lat = x-270;
    else
        %long = y;
        long = mod(y+90,360);
        lat = 90+x;
    end
    
end