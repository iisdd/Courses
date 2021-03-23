function [ xk] = tdsolve( Ak , Bk ,uk, x0 , kstart , kend )
% 求解线性时变离散系统状态方程
syms k
if(Bk == 0)
    Bk = zeros(size(Ak , 1) , 1);
end
xk = [];
for kk = kstart + 1: kend
    AA = eye(size (Ak));
    for i = kstart : kk - 1
        A = subs(Ak , 'k' , i);
        AA = A *AA;
    end
    AAB = eye(size(Ak));
    BB = zeros(size(Bk));
    for i = kk -1 : -1: kstart + 1
        A = subs(Ak , 'k' , i);
        AAB = AAB *A;
        B = subs(Bk , 'k' , kk -1-i +kstart);
        u = subs(uk , 'k' , kk - 1- i +kstart);
        BB = BB + AAB * B *u;
    end
    B = subs(Bk , 'k' , kk - 1);
    u = subs(uk , 'k' , kk -1);
    BB = BB + B * u;
    xk = [xk AA * x0 + BB];
end


end

