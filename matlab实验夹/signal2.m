%2.����һ���ȷֲ�����ֵΪ�㡢����Ϊ0.01��
%�������ź�u(n)
p = 0.01;N = 50000;
u = rand(1 , N);
u = u - mean(u);
a = sqrt(12 * p);u1 = u * a;
power_u1 = dot(u1 , u1)/N  %dot���ڻ�����var(u1)��Ч
plot(u1(1 : 100));grid on;



