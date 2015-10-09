function [pred] = networkPredict(data,theta,inputSize, hiddenSize,numClasses )
% Unroll the parameters from theta
[ W1,W2,b1,b2] = theta2params(theta,inputSize, hiddenSize,numClasses );
[n m] = size(data);%m为样本的个数，n为样本的特征数
%前馈
z2 = W1*data+repmat(b1,1,m);%注意这里一定要将b1向量复制扩展成m列的矩阵
a2 = sigmoid(z2);
z3 = W2*a2+repmat(b2,1,m);
a3 = sigmoid(z3);
%取最大
[prob,pred]=max(a3);
% ---------------------------------------------------------------------
end

