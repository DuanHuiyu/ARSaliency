function panoSalMap = calPanoSalMap(panoFixMap, imgCol)
% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% the value of sigma should change according to the application
% --------------------------------------------------------------------------
    sigmaEye = round(imgCol*3.34/360);
    window = fspecial('gaussian', 6*sigmaEye, sigmaEye);
    window = window/sum(sum(window));
    salMapEye = imfilter(panoFixMap, window, 'conv');
    panoSalMap = mat2gray(salMapEye);
end