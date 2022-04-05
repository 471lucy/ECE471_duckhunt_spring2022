function f = duckhunt_f_wdbc(w,D,mu) 

%the following is modified code from ECE 403 course material
%reference for this code: ECE 403 course material and lab code by Wu-Sheng Lu and Lei Zhao


%D = [Xtr; ytr]
%w is w hat (weight and bias)
y = D(1765,:); %y is the last row of 
x1 = D(1:1764,:); %x is row 1 to 30, all columns
x = [x1; ones(1,1300)];
%P is 285 in this case because we are using the training data
sum = 0; %for the summation in the softmax function
for p = 1:1:1300
a = log(1+exp(-y(p) * (w' * x(:,p)))); 
sum = sum + a;
end
f = (1/800)*sum + (mu/2)*norm(w)^2; %the summation of the softmax function, plus the 
%regularization term
end