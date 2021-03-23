%2.产生一均匀分布、均值为零、功率为0.01的
%白噪声信号u(n)
p = 0.01;N = 50000;
u = rand(1 , N);
u = u - mean(u);
a = sqrt(12 * p);u1 = u * a;
power_u1 = dot(u1 , u1)/N  %dot是内积。和var(u1)等效
plot(u1(1 : 100));grid on;



