clc;clear;
%% load data
datanames = {'glass','segment_uni', 'LM', 'USPSdata_20_uni', 'binalpha_uni', ...
    'ecoli_uni', 'CNAE-9', 'colon'};
data_idx = 8;
datapath = ['./data/', datanames{data_idx},'.mat'];
disp(['Loading dataset ', datanames{data_idx}]);
data = load(datapath);
X = data.X;Y = data.Y;
d = size(X,2);

%% 10-fold split data
disp(['10-fold data split']);
k = 10;
rng(0)
indices = crossvalind('Kfold', Y, k);
accs = zeros(k, 1);

%%
for i = 1:k
    % choose data for train and test
    [train_X, test_X, train_Y, test_Y] = one_fold_data(data, i, indices);
    
    % classification for each selected dimension
    model = fitcknn(train_X, train_Y, 'NumNeighbors', 1);
    predict_Y = predict(model, test_X);
    accs(i,1) = mean(test_Y == predict_Y);
end
acc = mean(accs,1);
disp(acc);
stds = std(accs);
