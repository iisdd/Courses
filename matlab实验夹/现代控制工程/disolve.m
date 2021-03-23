function [ Ak , AkBu] = disolve( A , B ,uz )
%������Զ�����ɢϵͳ״̬����
syms z
AA = z * eye(size(A)) - A;
invAA = inv(AA);
tAA = iztrans(invAA * z);
Ak = simplify(tAA);

if(B == 0)
    B = zeros(size(A ,1) , 1);
end
tAB = iztrans(invAA * B * uz);
AkBu = simplify(tAB);

end

