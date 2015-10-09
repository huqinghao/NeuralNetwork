%% STEP 0: Initialise constants and parameters
%
%  Here we define and initialise some constants which allow your code
%  to be used more generally on any arbitrary input. 
%  We also initialise some parameters used for tuning the model.
inputSize = 28*28;      % Size of input vector (MNIST images are 28x28)
numClasses = 10;        % Number of classes (MNIST images fall into 10 classes)
hiddenSize=300;
lambda = 1e-4; % Weight decay parameter

%%======================================================================
%% STEP 1: Load Data

load mnist;
inputData = images;


theta=networkTrain(inputSize, hiddenSize,numClasses, lambda, inputData, labels);   
pred=networkPredict(inputData,theta,inputSize, hiddenSize,numClasses);
acc = mean(labels(:) == pred(:));
fprintf('Train Accuracy: %0.3f%%\n', acc * 100);
%% Step 2: Test

inputData = testImages;

[pred] = networkPredict(inputData, theta,inputSize, hiddenSize,numClasses);

test_acc = mean(testLabels(:) == pred(:));
fprintf('Test Accuracy: %0.3f%%\n', acc * 100);


