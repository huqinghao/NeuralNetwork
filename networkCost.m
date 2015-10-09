function [cost, grad] = networkCost(theta,inputSize, hiddenSize, numClasses,lambda, data,labels)

[ W1,W2,b1,b2] = theta2params(theta,inputSize, hiddenSize,numClasses );

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% 

[n m] = size(data);%m为样本的个数，n为样本的特征数

groundTruth = full(sparse(labels, 1:m, 1));

%前向算法计算各神经网络节点的线性组合值和active值
z2 = W1*data+repmat(b1,1,m);%注意这里一定要将b1向量复制扩展成m列的矩阵
a2 = sigmoid(z2);
z3 = W2*a2+repmat(b2,1,m);
a3 = sigmoid(z3);

% 计算预测产生的误差
Jcost = (0.5/m)*sum(sum((a3-groundTruth).^2));

%计算权值惩罚项
Jweight = (1/2)*(sum(sum(W1.^2))+sum(sum(W2.^2)));

%损失函数的总表达式
cost = Jcost+lambda*Jweight;
%反向算法求出每个节点的误差值
d3 = -(groundTruth-a3).*sigmoidInv(z3);
                                                     
d2 = (W2'*d3).*sigmoidInv(z2); 
%计算W1grad 
W1grad = W1grad+d2*data';
W1grad = (1/m)*W1grad+lambda*W1;
%计算W2grad  
W2grad = W2grad+d3*a2';
W2grad = (1/m).*W2grad+lambda*W2;

%计算b1grad 
b1grad = b1grad+sum(d2,2);
b1grad = (1/m)*b1grad;%注意b的偏导是一个向量，所以这里应该把每一行的值累加起来

%计算b2grad 
b2grad = b2grad+sum(d3,2);
b2grad = (1/m)*b2grad;
%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

