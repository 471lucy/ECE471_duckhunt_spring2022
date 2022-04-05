function g = duckhunt_g_wdbc(w,D,mu)


%the following is modified code from ECE 403 course material
%reference for this code: ECE 403 course material and lab code by Wu-Sheng Lu and Lei Zhao

%gradient of softmax with the regularization term
y = D(1765,:); %y is the last row of 
x1 = D(1:1764,:); %x is row 1 to 30, all columns
x = [x1; ones(1,1300)];
%size(x)
%size(y)
sum = 0; %for the summation in the gradient of the softmax function
for p = 1:1:1300
a = (y(p)*x(:,p))/(1+exp(y(p) * (w' * x(:,p)))); 
sum = sum + a;
end
g = mu*w - (1/800)*sum; 
end