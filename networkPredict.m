function [pred] = networkPredict(data,theta,inputSize, hiddenSize,numClasses )
% Unroll the parameters from theta
[ W1,W2,b1,b2] = theta2params(theta,inputSize, hiddenSize,numClasses );
[n m] = size(data);%mΪ�����ĸ�����nΪ������������
%ǰ��
z2 = W1*data+repmat(b1,1,m);%ע������һ��Ҫ��b1����������չ��m�еľ���
a2 = sigmoid(z2);
z3 = W2*a2+repmat(b2,1,m);
a3 = sigmoid(z3);
%ȡ���
[prob,pred]=max(a3);
% ---------------------------------------------------------------------
end

