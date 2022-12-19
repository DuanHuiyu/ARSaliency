function train_test(model, testPath, featurePath, testImgs, features, featureSuffix, outputPath)

% model_liblinear = load(fullfile(params.path.data, 'model_liblinear.mat'));
model_liblinear = model;

if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end

meanVec = model_liblinear.whiteningParams(1, :);
stdVec = model_liblinear.whiteningParams(2, :);

outscale = 4;

for i = 1 :length(testImgs)
    fileName = testImgs{i,3};
%     X = collectFeatures(params, model_liblinear.model.testingImgs(j), model_liblinear.features);
    for j = 1 : length(features)
%         imgDir  = dir([params.path.maps.feature '/' features{j} '/*']);
%         name = imgDir(3).name;  % 1st is '.', 2nd name is '..'
%         postfix = name(end-3:end);
%         map = im2double(imread(fullfile(params.path.maps.feature, features{j}, [fileName(1:end-4) postfix])));
        temp_name = split(testImgs{i,j}, '.');
        map = im2double(imread(fullfile(featurePath, features{j}, [temp_name{1}, featureSuffix{j}])));
        if size(map,3)>1
            map = rgb2gray(map);
        end
        H = size(map,1)/outscale;
        W = size(map,2)/outscale;
        map = imresize(map, [H W]);
        allfeatures(1:size(map,1)*size(map,2), j) = map(:);
    end
    X = allfeatures;
    X = bsxfun(@minus, X, meanVec);
    X = X ./ repmat(stdVec, [size(X,1) 1]);
    
% %     MKL
%     Kt=mklkernel(X,model_liblinear.model.InfoKernel,model_liblinear.model.Weight,model_liblinear.model.options,model_liblinear.model.xapp,model_liblinear.model.beta);
%     predictions =Kt*model_liblinear.model.w+model_liblinear.model.b;
%     info = imfinfo(fullfile(params.path.stimuli, fileName));
%     W = info.Width/params.out.scale;
%     H = info.Height/params.out.scale;
%     map = reshape(predictions, [params.out.height params.out.width]);
%     map = reshape(predictions, [H W]);

% %   LIBSVM
%     testing_label_vector = rand(size(X,1),1);
%     [predicted_label, accuracy, prob_estimates] = svmpredict(testing_label_vector, X, model_liblinear.model.svm,  '-b 1');
%     info = imfinfo(fullfile(params.path.stimuli, fileName));
%     W = info.Width/params.out.scale;
%     H = info.Height/params.out.scale;
%     map = reshape(prob_estimates(:,1), [H W]);

%   LIBLINEAR
    predictions = X*model_liblinear.liblinear.w';
%     info = imfinfo(fullfile(params.path.stimuli, fileName));
%     W = info.Width/params.out.scale;
%     H = info.Height/params.out.scale;
    map = reshape(predictions, [H W]);
    
%     map = imfilter(map, params.out.gaussian);
    map = normalise(map);
    imwrite(map, fullfile(outputPath, fileName));
end

end

function [ normalised ] = normalise( map )
% [ normalised ] = normalise( map )
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

map = map - min(min(map));
s = max(max(map));
if s > 0
    normalised = map / s;
else
    normalised = map;
end

end