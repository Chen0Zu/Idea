function [train_X, test_X, train_Y, test_Y] = one_fold_data(data, i, indices)
test_idx = indices == i;
train_idx = ~test_idx;
train_X = data.X(train_idx,:);
test_X = data.X(test_idx,:);
train_Y = data.Y(train_idx,:);
test_Y = data.Y(test_idx,:);
end