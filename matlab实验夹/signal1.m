%1.����һ���ȷֲ��İ������ź�u(n)�������䲨�β�
%������ֲ����
clear
N = 50000;
u = rand(1, N);
u_mean = mean(u)
power_u = var(u)  %���ʵ��ڷ���
subplot(2 ,1, 1);
plot(u(1 : 100));grid on;
ylabel('u(n)');
subplot(2 , 1 , 2);
hist(u , 50);grid on;   %�ֳ�50�Σ�ÿ���ж��ٸ���������
ylabel('histogram of u(n)');
