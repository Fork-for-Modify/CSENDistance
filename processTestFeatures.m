%%
% Lei Zhang, Meng Yang, and Xiangchu Feng,
% "Sparse Representation or Collaborative Representation: Which Helps Face Recognition?", in ICCV 2011.

% The author of the modified implementation: Mete Ahishali
% Tampere University, Tampere, Finland.

% Zhihong Zhang
% 2021-7-8

%%
clc, clear, close all
addpath(genpath('crc'));

%% params
% Change the file name accordingly.
param.modelName = 'VGG19';
%1D or 2D corresponding to the traditional and proposed dictionary designs.
param.DicDesign = '2D';

% all data split to test
param.dictionary_size = 0; % Samples per class in the dictionary (0, all samples for test)
param.train_size = 0; % These are the proportations. 1:1
param.test_size = 1;


subdirname = 'test-parking-redgray/';
DicPath = './data/Dic/Dic2D.mat';
savedir = ['./results/' subdirname];
inputData = strcat('data/features/test-data-parking-redgray/features_max_', param.modelName, '.mat');
load(inputData)
objectFeatures = double(squeeze(objectFeatures));
angles = gtd(:, 1);
meters = gtd(:, 2);
outName = strcat(['data/split-feature/' subdirname], param.modelName);
if ~exist(['data/split-feature/' subdirname], 'dir')
   mkdir(['data/split-feature/' subdirname])
end

%% Pre-processing: Quantization and Sample Selection

% Samples between [0.5, 50.5] in meters. Quantization with 100 cms.
partition = 0.5:1:60.5;

codebook = zeros(length(partition) + 1, 1);
codebook(2:length(partition) + 1) = 1:length(partition);
codebook(1) = -1;
codebook(end) = -1;
[~, meters_quant] = quantiz(meters, partition, codebook);

% Remove out of range samples
objectFeatures(meters_quant == -1, :) = [];
meters(meters_quant == -1, :) = [];
meters_quant(meters_quant == -1, :) = [];

%% Collaborative Representation based Classification (CRC) implementation.

data = objectFeatures';
label = meters_quant;
reallabel = meters;

param.MR = 0.5; % Measurement rate.
param.k = 1;

measurement_type = 'eigen'; % Gauss, eigen, or None. None means no compression.
projection_matrix = 'l2_norm'; % minimum_norm or l2_norm.

% gen test sample and load dic
% [~, train, test] = split_data(data,label,param,reallabel);

%%
test.data = data;
test.reallabel = reallabel;
test.label = label;

train.data = zeros(512,0);
train.reallabel = zeros(0,1);
train.label = zeros(0,1);
%%

Dic = load(DicPath); Dic = Dic.Dic;

% Metrics.
% ard = zeros(1, nuR);
% srd = zeros(1, nuR);
% th = zeros(nuR, length(test(1).label));
% rmse = zeros(1, nuR);
% rmseLog = zeros(1, nuR);
% y_preds = zeros(nuR, length(test(1).label));
% y_trues = zeros(nuR, length(test(1).label));


% For the competing methods combine dictionary samples with training
% samples.
% xx_train = [Dic.dictionary'; train.data'];
% yy_train = [Dic.reallabel; train.reallabel];
xx_train = [];
yy_train = [];
xx_test = test.data';
yy_test = test.reallabel;

% CRC
[param.maskM, param.maskN] = size(Dic.label_matrix);
N = size(Dic.dictionary, 1); % Size of the feature vector.


dicRealLabel = Dic.reallabel; % Unquantized labels.
testRealLabel = test.reallabel;
trainRealLabel = train.reallabel;

D = Dic.dictionary; %This is the dictionary.

m = floor(param.MR * N); % number of measurements

% Dimensional reduction: measurement matrix Phi.
switch measurement_type
	case 'eigen'
		[phi,disc_value,Mean_Image]  =  Eigen_f(D,m);
		phi = phi';
	case 'Gauss'
		phi = randn(m, N);
	case 'None'
		m = 1;
		phi = 1;
		param.MR = 1;
end

A  =  phi*D;
A  =  A./( repmat(sqrt(sum(A.*A)), [m,1]) ); %normalization

% Measurements for dictionary.
Y0 = phi * Dic.dictionary;
energ_of_Y0 = sum(Y0.*Y0);
tmp = find(energ_of_Y0 == 0);
Y0(:,tmp)=[];
train.label(tmp) = [];
Y0 =  Y0./( repmat(sqrt(sum(Y0.*Y0)), [m,1]) ); %normalization

% Measurments for training.
Y1 = phi*train.data;
energ_of_Y1=sum(Y1.*Y1);
tmp=find(energ_of_Y1==0);
Y1(:,tmp)=[];
train.label(tmp)=[];
Y1 = Y1./( repmat(sqrt(sum(Y1.*Y1)), [m,1]) ); %normalization

% Measurments for test.
Y2 = phi*test.data;
energ_of_Y2=sum(Y2.*Y2);
tmp=find(energ_of_Y2==0);
Y2(:,tmp)=[];
test.label(tmp)=[];
Y2 = Y2./( repmat(sqrt(sum(Y2.*Y2)), [m,1]) ); %normalization

% Projection matrix computing
kappa             =   0.4; % l2 regularized parameter value
switch projection_matrix
	case 'minimum_norm'
		Proj_M=  pinv(A);
	case 'l2_norm'
		Proj_M = (A'*A+kappa*eye(size(A,2)))\A'; %l2 norm
	case 'transpose'
		Proj_M=  A';
end

%%%% Testing with CRC.
ID = [];
for indTest = 1:size(Y2,2)
	[id]    = CRC_RLS(A,Proj_M,Y2(:,indTest),Dic.label);
	ID      =   [ID id];
end

%%%%% Save variables
param.Proj_M = Proj_M;
param.Y0 = Y0;
param.Y1 = Y1;
param.Y2 = Y2;
param.trainLabel = train.label;
param.testLabel = test.label;
param.dicRealLabel = dicRealLabel;
param.trainRealLabel = trainRealLabel;
param.testRealLabel = testRealLabel;

% Compute necessary variables for CSEN training and testing.
prepareCSEN(subdirname,Dic, param);	
disp('CSEN preprocessing data saved');

% save and show CRC result
save(strcat(outName, '_mr_', num2str(param.MR), ...
	'_run', num2str(param.k), ('.mat')), ...
	'xx_train', 'xx_test', 'yy_train', 'yy_test', '-v6')

ard = sum(abs(ID' - test.reallabel)./test.reallabel) ...
			/ length(test.reallabel);
srd = sum(((ID' - test.reallabel).^2)./test.reallabel) ...
			/ length(test.reallabel);
th = max(test.reallabel./ ID', ID'./test.reallabel);
rmse = sqrt(sum((ID' - test.reallabel).^2) / length(test.reallabel));
rmseLog = sqrt(sum((log(ID') - log(test.reallabel)).^2) ...
			/ length(test.reallabel));

y_trues = test.reallabel;
y_preds = ID';


outName_results = strcat(['results/' subdirname 'CRC_base/'] , param.modelName);
if ~exist(['results/' subdirname 'CRC_base/'], 'dir')
mkdir(['results/' subdirname 'CRC_base/'])
end
save([outName_results '_pred.mat'], 'y_trues', 'y_preds');
disp('CRC results saved');

figure,
scatter(y_trues, y_preds, 3, 'filled')
axis([0,60,0,60])
title(strcat('Collaborative Filtering, MSE: ', ...
num2str(sum((y_trues- y_preds).^2)/length(y_trues))))
xlabel('Actual Distance in meters'), ylabel('Predicted Distance in meters')
