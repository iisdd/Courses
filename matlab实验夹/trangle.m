a = input('������߳�1:');
b = input('������߳�2:');
c = input('������߳�3:');

if (a + b > c) && (a + c > b) &&(b + c > a)
    fprintf('yes\n')
else
    fprintf('nope\n')
end
% if a + b <= c   
%     fprintf('no\n')
% elseif a + c <= b
%     fprintf('no\n')
% elseif b + c <= a
%     fprint('no\n')
% else
%     fprintf('yes\n')
% end
