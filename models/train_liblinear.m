function model = train_liblinear(X, Y)

%%%%%%%%%%%%
% Training %
%%%%%%%%%%%%

c=1; % parameter for the liblinear machine learning 

% normalise the training data
meanVec=mean(X, 1);
X=X-repmat(meanVec, [size(X, 1), 1]);

stdVec=std(X);
stdVec(stdVec==0) = 1;
X=X./repmat(stdVec, [size(X, 1), 1]);

whiteningParams = [meanVec; stdVec];


Y(Y==0) = -1;

liblinear_params=['-s 1 ', '-c', blanks(1), num2str(c), ' -B -1'];
model.liblinear=train(Y, sparse(X), liblinear_params);

model.whiteningParams = whiteningParams;




end
