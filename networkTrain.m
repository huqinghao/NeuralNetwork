function [opttheta] = networkTrain(inputSize, hiddenSize,numClasses, lambda, inputData, labels)

%% ³õÊ¼»¯  Obtain random parameters theta
theta = initializeParameters(inputSize, hiddenSize,numClasses);

%% ×îÓÅ»¯ËÑË÷
%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 200;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

[opttheta, cost] = minFunc( @(p) networkCost(p,inputSize, hiddenSize, numClasses,lambda, inputData,labels), ...
                              theta, options);
save opttheta.mat opttheta;
end
