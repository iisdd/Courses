%1.14 ��x(n) =   Asin(wn) + u(n),����w = pi/16,u(n)�ǰ�������
%1.������ֵΪ0������p1 = 0.1�ľ��ȷֲ�������u(n)��������ͼ�Σ�����u(n)������غ���ru(m)������ru(m)�Ĳ��Ρ�
N =50000;p1 = 0.1; a = sqrt(p1 * 12);Mlag = 50;
u = rand(1 , N);
u = u - mean(u);u = a * u ;nu = 1:100;
power_u = var(u)
subplot(2 , 2 , 1);
plot(u(nu));grid on ; title('������u(n)') ; xlabel('n') ;ylabel('u(n)');
ru = xcorr(u , Mlag , 'biased');nru = -Mlag:Mlag;
subplot(2 , 2, 2);
plot(nru , ru);grid on;title('u����غ���ru(m)');xlabel('m') ; ylabel('ru(m)');
%2.��ʹx(n)�������Ϊ10dB���Ծ���A����ֵ��������x(n)��ͼ�μ�������غ���rx(m)��ͼ�Ρ�
f = pi/16;A = sqrt(2);n = (0 : N-1);
s = A *sin(2 * pi* f* n);
x = s + u(1 : N);nx = 1:100;
power_s = var(s)  % lg(p2/p1) = 10,p1 = 0.1,����s�Ĺ���Ϊ1��
subplot(2 , 2 ,3);
plot(x(nx));grid on ;title('x�ź�');xlabel('n');ylabel('x(n)');
rx = xcorr(x , Mlag,'biased');nrx = -Mlag : Mlag;
subplot(2 , 2, 4);
plot(nrx , rx);grid on ; title('x����غ���rx(m)');xlabel('m');ylabel('rx(m)');


