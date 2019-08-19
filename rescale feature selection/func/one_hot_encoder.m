function encoder = one_hot_encoder(Y)
I = eye(max(Y));
encoder = I(Y,:);
end