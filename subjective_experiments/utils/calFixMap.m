function panoFixMap = calFixMap(ImgRow, ImgCol, Lat, Long)
% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
    black = zeros(ImgRow,ImgCol);
    for i = 1:size(Lat,1)
        for j = 1:size(Lat,2)
            if isnan(Lat(i,j)) || isnan(Long(i,j)) || (Lat(i,j)==1&&Long(i,j)==1)
                continue
            end
            black(Lat(i,j), Long(i,j)) = 1;
        end
    end
    panoFixMap = black;

end