% 让用户输入一个数字num
% 这个数字num必须是1 - 100之间的数字
% 如果不在这个范围内，则输出'wrong number'
% 如果数字在这个范围内，则输出这个数字的平方
num = input('请输入一个1 - 100之间的整数');
% if num <= 100 && num >= 1
%     disp(num^2)
% else
%     fprintf('wrong number')
% end
if num < 1 || num >100
    fprintf('wrong number')
else
    disp(num^2)
end