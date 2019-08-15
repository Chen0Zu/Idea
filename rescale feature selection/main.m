
clc;clear;
%% load data
datanames = {'glass','segment_uni', 'LM', 'USPSdata_20_uni', 'binalpha_uni', ...
    'ecoli_uni', 'CNAE-9', 'colon'};
datapath = './data/glass.mat';
disp(['Loading dataset ', dataname]);
data = load(datapath);
X = data.X;Y = data.Y;

%% 10-fold split data
disp(['10-fold data split']);
k = 10;
indices = crossvalind('Kfold', Y, k);

%%
for i = 1:k
    [train_X, test_X, train_Y, test_Y] = one_fold_data(data, i, indices);
    gamma = 1;
    max_iter = 10;
    max_iter2 = 10;
    [ ranked, theta,W,obj] = ...
        scalefs( train_X', one_hot_encoder(train_Y));
end

