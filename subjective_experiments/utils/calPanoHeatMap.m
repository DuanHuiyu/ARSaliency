function panoHeatMap = calPanoHeatMap(img, panoSalMap)
% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
    [ImgRow,ImgCol,~] = size(panoSalMap);
    [meshX, meshY] = meshgrid(1:ImgCol, 1:ImgRow);
    figure
    imshow(img)
    hold on
    panoHeatMap = pcolor(meshX,meshY,panoSalMap);
    colorbar;
    shading interp
    alpha(0.4)
    hold off
    % axis([7800 9000 2200 3500]);
end