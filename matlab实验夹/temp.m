clear
N = 500;Mlag = 50;
nx = 0:N-1;
x = exp(-nx*0.1);
rx = xcorr(x , Mlag , 'biased');
nrx = -Mlag :Mlag;
subplot(2 , 1 ,1);plot(nx, x);grid on;
subplot(2 ,1 ,2);plot(nrx , rx);grid on;