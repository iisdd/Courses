%1.14 令x(n) =   Asin(wn) + u(n),其中w = pi/16,u(n)是白噪声。
%1.产生均值为0，功率p1 = 0.1的均匀分布白噪声u(n)，画出其图形，并求u(n)的自相关函数ru(m)，画出ru(m)的波形。
N =50000;p1 = 0.1; a = sqrt(p1 * 12);Mlag = 50;
u = rand(1 , N);
u = u - mean(u);u = a * u ;nu = 1:100;
power_u = var(u)
subplot(2 , 2 , 1);
plot(u(nu));grid on ; title('白噪声u(n)') ; xlabel('n') ;ylabel('u(n)');
ru = xcorr(u , Mlag , 'biased');nru = -Mlag:Mlag;
subplot(2 , 2, 2);
plot(nru , ru);grid on;title('u自相关函数ru(m)');xlabel('m') ; ylabel('ru(m)');
%2.欲使x(n)的信噪比为10dB，试决定A的数值，并画出x(n)的图形及其自相关函数rx(m)的图形。
f = pi/16;A = sqrt(2);n = (0 : N-1);
s = A *sin(2 * pi* f* n);
x = s + u(1 : N);nx = 1:100;
power_s = var(s)  % lg(p2/p1) = 10,p1 = 0.1,所以s的功率为1。
subplot(2 , 2 ,3);
plot(x(nx));grid on ;title('x信号');xlabel('n');ylabel('x(n)');
rx = xcorr(x , Mlag,'biased');nrx = -Mlag : Mlag;
subplot(2 , 2, 4);
plot(nrx , rx);grid on ; title('x自相关函数rx(m)');xlabel('m');ylabel('rx(m)');


