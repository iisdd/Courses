function [ Phi , PhiBu ] = tsolve(A , B ,u ,x , a , n )
%求解时变系统状态方程
Phi = transmtx(A , x , a , n);
Phitao = subs(Phi , x ,'tao');
if(B == 0)
    Btao = zeros(size(A , 1) , 1);
else
    Btao = subs(B , x , 'tao');
end
utao = subs(u , x , 'tao');
PhiBu = simplify(int(Phitao * Btao * utao , 'tao' , a ,x));

end

