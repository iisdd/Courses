a = input('a = ');
b = input('b = ');
c = input('c = ');

delta = b^2 - 4 * a * c;
if delta > 0
    x1 = (-b + sqrt(delta))/2*a
    x2 = (-b - sqrt(delta))/2*a
elseif delta  == 0
    x = -b/(2*a)
  
else
    fprintf('no solution\n');
end