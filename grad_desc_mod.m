
%the following is modified gradient descent code provided from ECE 403 course material
%reference for this code: ECE 403 course material and lab code by Wu-Sheng Lu and Lei Zhao



%To implement the modified gradient descent algorithm: [xs,fs,k] =
%[ws, fs, k] = grad_desc_mod('f_wdbc', 'g_wdbc',zeros(31,1), K, D, mu);

% To implement the gradient descent algorithm.
% Example: [xs,fs,k] = grad_desc('f_rosen','g_rosen',[0; 2],1e-9);
function [xs,fs,k] = grad_desc_mod(fname,gname,x0,K, D, mu) %modified for D and mu are inputs, changed epsi to K for terminating
%fprintf('here')
format compact
format long
k = 1;
xk = x0;
gk = feval(gname,xk,D,mu); %modified to include D, and mu
dk = -gk;
ak = bt_lsearch2019(xk,dk,fname,gname,D,mu); %modified to include D, and mu
adk = ak*dk;
er = norm(adk);
while k < K % modified so it ends after K iterations
  xk = xk + adk;
  gk = feval(gname,xk,D,mu); %modified to include D and mu
  dk = -gk;
  %fprintf('here')
  ak = bt_lsearch2019(xk,dk,fname,gname,D,mu); %modified to include D, and mu
  adk = ak*dk;
  er = norm(adk);
  k = k + 1;
end
disp('solution:')
xs = xk + adk
disp('objective function at solution point:')
fs = feval(fname,xs,D,mu) %modified to include D and mu
format short
disp('number of iterations performed:')
k