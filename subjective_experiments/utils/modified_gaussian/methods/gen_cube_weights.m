function [ equi_weights ] = gen_cube_weights( new_imgH )
%GEN_CUBE_WEIGHTS Generates weights based on center of cube face for the
%specified size. Returns an equirectangular image with cube weights

    %using parameters from
    %Thomas Maugey, Olivier Le Meur, and Zhi Liu. 2017. Saliency-based navigation
%     in omnidirectional image. In Multimedia Signal Processing (MMSP), 2017 IEEE
%     19th International Workshop on. IEEE, 1–6.
    q = 10;
    d_0 = 0.3*new_imgH;

    fov = pi /2;          % horizontal angle of view
    VectX = [ 0 -pi -pi/2 0 pi/2  0];
    VectY = [ pi/2 0 0 0 0  -pi/2 ];
    
    %gen weights on square img
    %weights = zeros(new_imgH);
    a= linspace(-floor(new_imgH/2),floor(new_imgH/2),new_imgH);
    b= linspace(-floor(new_imgH/2),floor(new_imgH/2),new_imgH);
    
    [a_mesh,b_mesh] = meshgrid(a,b);
    %d_ij = max(a_mesh,b_mesh).^2;
    d_ij = max(a_mesh.^2,b_mesh.^2);
    weights = 1./(1+(d_ij./d_0)^q);
    
    
    equi_weights = zeros(new_imgH*2,new_imgH*4);
    for i = 1:6
        [sphereImg_temp, validMap_temp] = im2Sphere(weights, fov, 4*new_imgH, 2*new_imgH, VectX(i), VectY(i));
        equi_weights(validMap_temp) = sphereImg_temp(validMap_temp);
    end

end

