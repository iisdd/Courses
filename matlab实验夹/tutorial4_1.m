% ���û�����һ������num
% �������num������1 - 100֮�������
% ������������Χ�ڣ������'wrong number'
% ��������������Χ�ڣ������������ֵ�ƽ��
num = input('������һ��1 - 100֮�������');
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