a = input('a = ');
b = input('b = ');
r = mod(a , b);
while r ~= 0
    a = b;
    b = r;
    r = mod(a , b);
end
fprintf('���Լ��Ϊ��%g\n', b)