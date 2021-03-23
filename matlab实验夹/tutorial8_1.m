x1 =   -5  : 0.1  : 5;
y1 =  x1.^ 2;
plot (x1 , y1);
hold on
x2 = -5 : 0.1 : 5;
y2 = x2 .^ 3 ;
plot(x2 , y2);
grid on
title('x^2 vs x^3');
xlabel('x-axis');
ylabel('y-axis');
