x = -4 : 0.1 : 4;
y1 = sin(x);
y2 = sin(2 .* x);
y3 = cos(3 .* x);
subplot(2 , 2 , 1);
plot(x , y1);
title('y = sinx')
subplot(2 , 2 , 2 );
plot(x , y2);
title('y = sin2x')
subplot(2 , 2 , [3 4]);
plot(x , y3);
title('y = cos3x')