clc;clear;
sigmas = [0.1, 1, 10];
x = -3:0.1:3;
l1 = abs(x);
l2 = x.^2;
lsigma01 = (1+sigmas(1))*x.^2./(abs(x)+sigmas(1));
lsigma1 = (1+sigmas(2))*x.^2./(abs(x)+sigmas(2));
lsigma10 = (1+sigmas(3))*x.^2./(abs(x)+sigmas(3));

figure;
hold on;
linewidth = 2;
plot(x,l1, 'linewidth', linewidth);
plot(x, l2,'linewidth', linewidth);
plot(x, lsigma01,'linewidth', linewidth);
box on;
h = legend('$\ell_1$-norm', '$\ell_2$-norm', 'Adaptive-loss');
set(h,'Interpreter','latex','Location','Best', 'Fontsize', 18);

figure;
hold on;
linewidth = 2;
plot(x,l1, 'linewidth', linewidth);
plot(x, l2,'linewidth', linewidth);
plot(x, lsigma1,'linewidth', linewidth);
box on;
h = legend('$\ell_1$-norm', '$\ell_2$-norm', 'Adaptive-loss');
set(h,'Interpreter','latex','Location','Best', 'Fontsize', 18);

figure;
hold on;
linewidth = 2;
plot(x,l1, 'linewidth', linewidth);
plot(x, l2,'linewidth', linewidth);
plot(x, lsigma10,'linewidth', linewidth);
box on;
h = legend('$\ell_1$-norm', '$\ell_2$-norm', 'Adaptive-loss');
set(h,'Interpreter','latex','Location','Best', 'Fontsize', 18);


