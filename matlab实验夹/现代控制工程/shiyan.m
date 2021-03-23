
A = [-1 -2 -2 ;0 -1 1;1 0 -1];
B = [2;0;1];
C = [1 1  0];
p = [-1 -1 -3];p0 = [-1 -1 -3];
G = acker(A' , C' , p0)
K = acker(A , B, p)
A0 = [[A - B * K;G' * C ] [ -B * K ;A - G' * C - B * K]]
SysPole = eig(A0)