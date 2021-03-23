%利用sinc.m文件产生一个sinc信号并显示其波形
%sinc(t) = sin(t)/t
n  = 200;
stept = 4*pi /n;
t = -2 * pi:stept:2*pi;
y = sinc(t);
plot(t , y,t, zeros(size(t)));grid on;
%相当于：
%plot(t , y);
% hold on;
% plot(t, zeros(size(t)));