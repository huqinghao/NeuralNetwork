function [opttheta] = networkTrain(inputSize, hiddenSize,numClasses, lambda, inputData, labels)

%% 初始化  Obtain random parameters theta
theta = initializeParameters(inputSize, hiddenSize,numClasses);

%% 最优化搜索
%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 10;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

[opttheta, cost] = minFunc( @(p) networkCost(p,inputSize, hiddenSize, numClasses,lambda, inputData,labels), ...
                              theta, options);
save opttheta.mat opttheta;
end
