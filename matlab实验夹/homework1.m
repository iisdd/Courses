%例题1.8.3
N = 500; p1 = 1; p2 = 0.1;f = 1/8;Mlag = 60;
u =randn(1 , N); 
a = sqrt(p2); u1 = a*u ;
n = (0:N-1);
s = sin(2 * pi* f* n);
x1 = u(1 :N) + s;
rx1 = xcorr(x1 , Mlag , 'biased');nrx1 = (-Mlag : Mlag);
x2 = u1(1 : N) + s;
rx2 = xcorr(x2 , Mlag , 'biased');nrx2 = (-Mlag : Mlag);
subplot(2 ,2 , 1);plot( x1(1:Mlag));axis([0 , 60 , -2.5 , 2.5]);set(gca,'XTick',[0:10:60]);set(gca,'YTick',[-2.5:0.5:2.5]);
title('x1信号');xlabel('n');ylabel('x1(n)');
subplot(2 , 2, 2);plot(nrx1 , rx1);axis([0 , 60, -1.0 , 2.0]);set(gca,'XTick',[0:10:60]);set(gca,'YTick',[-1.0:0.5:2.0]);
title('x1信号自相关');xlabel('m');ylabel('rx1(m)');
subplot(2 ,2 ,3); plot(n , x2);axis([0 , 60 , -1.5 , 1.5]);set(gca,'XTick',[0:10:60]);set(gca,'YTick',[-1.5:0.5:1.5]);
title('x2信号');xlabel('n');ylabel('x2(n)');
subplot(2 , 2,4); plot(nrx2 , rx2);axis([0 , 60 , -0.6 , 0.8]);set(gca,'XTick',[0:10:60]);set(gca,'YTick',[-0.6:0.2:0.8]);
title('x2信号自相关');xlabel('m');ylabel('rx2(m)');