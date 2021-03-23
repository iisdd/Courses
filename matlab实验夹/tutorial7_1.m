x = -2 :0.1: 2;
y1 = x .^2;
y2 = x .^3;
%plot(x , y1 , 'green - o')
%plot(x , y1 , 'green' , x  ,y2 , 'black')
y = [75 91 105 123.5 131 150 179 203 196 350];

x1 = 2000 : 2009;
bar(x1 , y)