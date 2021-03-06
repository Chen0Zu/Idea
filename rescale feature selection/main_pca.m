clc;clear;
addpath('./func')
%% load data
datanames = {'glass','segment_uni', 'LM', 'USPSdata_20_uni', 'binalpha_uni', ...
    'ecoli_uni', 'CNAE-9', 'colon'};
data_idx = 8;
datapath = ['./data/', datanames{data_idx},'.mat'];
disp(['Loading dataset ', datanames{data_idx}]);
data = load(datapath);
X = data.X;Y = data.Y;
d = size(X,2);
feas = ceil(50*[1:10]*0.1);
%% 10-fold split data
disp(['10-fold data split']);
k = 10;
rng(0)
indices = crossvalind('Kfold', Y, k);
accs = zeros(k, length(feas));
%%
for i = 1:k
    % choose data for train and test
    [train_X, test_X, train_Y, test_Y] = one_fold_data(data, i, indices);
    
    % feature selection
    [coeff, transformed_train_X, latent, tsquared, explained, mu] = pca(train_X);
    % classification for each selected dimension
    for j = 1:length(feas)
        model = fitcknn(transformed_train_X(:,1:feas(j)), train_Y, 'NumNeighbors', 1);
        transformed_test_X = bsxfun(@minus, test_X, mu)*coeff;
        predict_Y = predict(model, transformed_test_X(:,1:feas(j)));
        accs(i,j) = mean(test_Y == predict_Y);
    end
end
acc = mean(accs,1);
stds = std(accs);
disp(acc);

