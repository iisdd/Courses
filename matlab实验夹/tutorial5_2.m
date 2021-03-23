s = 0;
for i = 1 : 100
    if ~mod(i , 2) 
        s = s - 1/i ;
    else
        s = s + 1/i;
    end
        
end
disp(s)