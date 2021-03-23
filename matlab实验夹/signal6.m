%令x(n) = {1, 2, 3, 4 ,5},h(n) = {6, 2, 3, 6, 4, 2},
%y(n) = x(n) * h(n)(卷积)求y(n)
N = 5; M = 6; L = N + M - 1;
x = [1 , 2 , 3 ,4 ,5];nx = 0: N-1;
h = [6, 2 ,3 ,6 ,4 ,2];nh = 0: M-1;
y = conv(x , h);ny = 0:L-1;
%stem用来绘制离散序列的图形
subplot(2, 2 ,1);stem(nx , x , 'x');xlabel('n');ylabel('x(n)');grid on;
subplot(2 ,2, 2);stem(nh ,h , 'o');xlabel('n');ylabel('h(n)');grid on;
subplot(2 ,2 ,[3 4]);stem(ny , y , '.');xlabel('n');ylabel('y(n)');grid on;

