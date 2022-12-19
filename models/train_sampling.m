function [X Y] = train_sampling(trainingPath, featurePath, trainingImgs, features, featureSuffix)
% [X Y] = sampling(params, trainingImgs, features)
%
% ----------------------------------------------------------------------
% Matlab tools for "Saliency in crowd," ECCV, 2014
% Ming Jiang, Juan Xu, Qi Zhao
%
% Copyright (c) 2014 NUS VIP - Visual Information Processing Lab
%
% Distributed under the MIT License
% See LICENSE file in the distribution folder.
% -----------------------------------------------------------------------
outscale = 4;

posPtsPerImg=10; % number of positive samples taken per image to do the learning
negPtsPerImg=10; % number of negative samples taken
p=20; % pos samples are taken from the top p percent salient pixels of the fixation map
q=30; % neg samples are taken from below the top q percent

% H = params.out.height;
% W = params.out.width;
% featuresTraining = collectFeatures(params, trainingImgs, features); % this should be size [M*N*numImages, numFeatures]
% labels = zeros(length(trainingImgs) * H * W, 1);

for i = 1 : length(trainingImgs)
    fileName = trainingImgs{i,3};
    map = im2double(imread([trainingPath, fileName]));
    H = size(map,1)/outscale;
    W = size(map,2)/outscale;
    map = imresize(map, [H W]);
    
    imageLabels=im2single(map);
    % eliminate the borders
    border = 7; %num of pixels along the border that one should not choose from.
    imageLabels(1:border, :) = -1;
    imageLabels(end-border+1:end, :) = -1;
    imageLabels(:, 1:border) = -1;
    imageLabels(:, end-border+1:end) = -1;
    
    imageLabels=reshape(imageLabels, [W*H, 1]);
    [A, IX] = sort(imageLabels, 'descend');
    
    % Find the positive examples in the top p percent
    ii = ceil((p/100)*length(imageLabels)*rand([posPtsPerImg, 1]));
    posIndices = IX(ii);
    
    % Find the negative examples from below top q percent
    % in practice, we find indices between [a, b]
    a = (q/100)*length(imageLabels); % top q percent
    b = length(imageLabels)-length(find(imageLabels==-1)); % index before border
    jj = ceil(a + (b-a).*rand(negPtsPerImg,1));
    negIndices = IX(jj);
    
    % map_index = zeros(W*H,1);
    % map_index(posIndices) = 1;
    % imwrite(reshape(map_index,[H W]),['./data/train_points/' fileName(1:end-4) '_p.jpg']);
    % map_index = zeros(W*H,1);
    % map_index(negIndices) = 1;
    % imwrite(reshape(map_index,[H W]),['./data/train_points/' fileName(1:end-4) '_n.jpg']);
    
    for j = 1 : length(features)
        temp_name = split(trainingImgs{i,j}, '.');
        map = im2double(imread(fullfile(featurePath, features{j}, [temp_name{1}, featureSuffix{j}])));
        if size(map,3)>1
            map = rgb2gray(map);
        end
        map = imresize(map, [H W]);
        map = reshape(map, [H*W, 1]);
        trainingFeaturesPos( (i-1)*posPtsPerImg+1 : i*posPtsPerImg ,j) = map(posIndices);
        trainingFeaturesNeg( (i-1)*negPtsPerImg+1 : i*negPtsPerImg ,j) = map(negIndices);
    end
end
X = double([trainingFeaturesPos; trainingFeaturesNeg]); %trainingFeatures
Y = double([ones(1, size(trainingFeaturesPos,1)), zeros(1, size(trainingFeaturesPos,1))])'; %trainingLabels

end