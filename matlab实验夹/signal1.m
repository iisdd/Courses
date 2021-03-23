%1.产生一均匀分布的白噪声信号u(n)，画出其波形并
%检验其分布情况
clear
N = 50000;
u = rand(1, N);
u_mean = mean(u)
power_u = var(u)  %功率等于方差
subplot(2 ,1, 1);
plot(u(1 : 100));grid on;
ylabel('u(n)');
subplot(2 , 1 , 2);
hist(u , 50);grid on;   %分成50段，每段有多少个落在里面
ylabel('histogram of u(n)');
