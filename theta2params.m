function [ W1,W2,b1,b2] = theta2params(theta,inputSize, hiddenSize,numClasses )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
W1 = reshape(theta(1:hiddenSize*inputSize), hiddenSize, inputSize);
W2 = reshape(theta(hiddenSize*inputSize+1:hiddenSize*inputSize+hiddenSize*numClasses), numClasses, hiddenSize);
b1 = theta(hiddenSize*inputSize+hiddenSize*numClasses+1:hiddenSize*inputSize+hiddenSize*numClasses+hiddenSize);
b2 = theta(hiddenSize*inputSize+hiddenSize*numClasses+hiddenSize+1:end);

end

