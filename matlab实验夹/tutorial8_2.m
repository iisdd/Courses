x = -4 : 0.1 : 4;
y1 = sin(x);
y2 = sin(2 .* x);
y3 = sin(3 .* x);
y4 = sin(4 .* x);

subplot(2 , 2 , 1);
plot(x , y1);
title('y = sinx')
subplot(2 , 2 , 2 );
plot(x , y2);
title('y = sin2x')
subplot(2 , 2 , 3);
plot(x , y3);
title('y = sin3x')
subplot(2 , 2 , 4);
plot(x , y4)
title('y = sin4x')

