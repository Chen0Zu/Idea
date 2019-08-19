clc;clear;
%% load data
datanames = {'glass','segment_uni', 'LM', 'USPSdata_20_uni', 'binalpha_uni', ...
    'ecoli_uni', 'CNAE-9', 'colon'};
data_idx = 1;
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
accs = zeros(k, d);
%%
for i = 1:k
    % choose data for train and test
    [train_X, test_X, train_Y, test_Y] = one_fold_data(data, i, indices);
    
    % feature selection
    
    % classification for each selected dimension
    for j = 1:d
        model = fitcknn(train_X(:,ranked(1:j)), train_Y, 'NumNeighbors', 1);
        predict_Y = predict(model, test_X(:,ranked(1:j)));
        accs(i,j) = mean(test_Y == predict_Y);
    end
end
acc = mean(accs,1);
disp(acc);

