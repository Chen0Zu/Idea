clc;clear;
addpath('./FSLib_v6.1_2018/lib'); % dependencies
addpath('./FSLib_v6.1_2018/methods'); % FS methods
%% load data
datanames = {'glass','segment_uni', 'LM', 'USPSdata_20_uni', 'binalpha_uni', ...
    'ecoli_uni', 'CNAE-9', 'colon'};
data_idx = 8;
datapath = ['./data/', datanames{data_idx},'.mat'];
disp(['Loading dataset ', datanames{data_idx}]);
data = load(datapath);
X = data.X;Y = data.Y;
d = size(X,2);
feas = ceil(d*[1:10]*0.1);
%% 10-fold split data
disp(['10-fold data split']);
k = 10;
rng(0)
indices = crossvalind('Kfold', Y, k);
accs = zeros(k, length(feas));
%%
for i = 1:k
    % choose data for train and test
    [X_train, X_test, Y_train, Y_test] = one_fold_data(data, i, indices);
    
    % feature selection
    W = dist(X_train');
    W = -W./max(max(W)); % it's a similarity
    [lscores] = LaplacianScore(X_train, W);
    [junk, ranked] = sort(-lscores);
    % classification for each selected dimension
    for j = 1:length(feas)
        model = fitcknn(X_train(:,ranked(1:feas(j))), Y_train, 'NumNeighbors', 1);
        predict_Y = predict(model, X_test(:,ranked(1:feas(j))));
        accs(i,j) = mean(Y_test == predict_Y);
    end
end
acc = mean(accs,1);
disp(acc);
stds = std(accs);
disp(stds)

