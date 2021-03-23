%3.产生零均值、功率为0.1且服从
%高斯分布(正态分布)的白噪声信号u(n)
p = 0.1;
N = 500000;
a = sqrt(p);
u = randn(1 , N);
u1 = a * u;
power_u1 = var(u1)
subplot(2 , 1 ,1);
plot(u1(1 : 100));grid on;
subplot(2 , 1 ,2);
hist(u , 50);