t = 0 :pi / 20: 2 * pi;
X = cos(t);
Y = sin(t);

plot(X , Y);
hold on 
axis equal
x = [0 ,1];
y = [0 ,0];
h = plot(x , y);
theta = 0;
while 1
    theta = theta + 0.01;
    x(2) = cos(theta);
    y(2) = sin(theta);

    set(h , 'XData' ,x , 'YData' , y)
    drawnow
end