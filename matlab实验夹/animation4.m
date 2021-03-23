X = -10 : 0.4 : 10;
Y = -10 : 0.4 : 10;

[XX , YY] = meshgrid(X, Y);
ZZ = sin(XX) + cos(YY);

h = surf(XX, YY ,ZZ);
axis([-10 , 10 , -10 , 10 , -5 ,5]);

while 1 
    for i = 1: 100;
        ZZ = 0.97 * ZZ;
        set(h , 'XData' , XX , 'YData' , YY , 'ZData' , ZZ);
        drawnow
    end
    for i = 1:100;
        ZZ = ZZ /0.97;
        set(h , 'XData' , XX , 'YData' , YY , 'ZData' , ZZ);
        drawnow
    end
end