%����sinc.m�ļ�����һ��sinc�źŲ���ʾ�䲨��
%sinc(t) = sin(t)/t
n  = 200;
stept = 4*pi /n;
t = -2 * pi:stept:2*pi;
y = sinc(t);
plot(t , y,t, zeros(size(t)));grid on;
%�൱�ڣ�
%plot(t , y);
% hold on;
% plot(t, zeros(size(t)));