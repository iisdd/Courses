function [ Phi ] = transmtx( A , x ,a ,n )
%���ʱ��ϵͳ״̬ת�ƾ���
Phi = eye(size(A));
for lop = 0 : n
    AA = A;
    for i = 1 : lop
        if(AA == 0)
            break
        end
        Atemp = subs(AA , x ,'tao');
        AA =simplify(A *int(Atemp , 'tao' ,a ,x));
    end
    if(AA == 0)
        break;
    end
    Atemp = subs(AA , x,'tao');
    AA = simplify(int(Atemp , 'tao' ,a ,x));
    Phi = simplify(Phi + AA);
end


end

