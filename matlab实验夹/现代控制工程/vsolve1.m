function [ Phit , PhitBu ] = vsolve1( A , B ,ut )
%����ָ������״̬����
syms t tao 
Phit = expm(A * t);
if(B == 0)
    B = zeros(size(A ,1) , 1);
end
phi = subs(Phit , 't' , 't - tao');
PhitBu = int(phi * B * ut , 'tao' , 0 , 't');

end

