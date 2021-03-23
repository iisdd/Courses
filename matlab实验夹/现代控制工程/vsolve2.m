function [sI_A , sI_ABu ] = vsolve2( A, B,us )
%拉式变换法
syms s
AA = s * eye(size(A)) - A;
invAA = inv(AA) ; 
tAA = ilaplace(invAA);
sI_A = simplify(tAA);
if(B == 0)
    B = zeros(size(A , 1) , 1);
end
tAB = ilaplace(invAA * B * us);
sI_ABu = simplify(tAB);


end

