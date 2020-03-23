% demo for updating singular value decomposition

clc;clear; close all;

m = 500; n = 50;
window = 20;
kappa = 1.0e+5;
epsilon = 0.1;

X = geometric(m,n, kappa);

Xi = X(:,1:n-window); 
xdata = X(:,(n-window+1):end);  

[U, R, V] = svd(Xi, 'econ');
if epsilon == 0
    epsilon = sqrt(sum(diag(R).^2)) * 0.1;
end
%  truncation..0
%     [idx, ~] = find(diag(R) < epsilon / 5);
sum = 0; max_sum = norm(R, 'fro')^2;
for idx = 1:size(R,1)
    sum = sum + R(idx, idx)^2;
    if sqrt(max_sum - sum) < epsilon * 0.9
        break;
    end
end
Us = U(:,1:idx); Rs = R(1:idx, 1:idx); Vs = V(:, 1:idx);
normE = norm(Xi - Us * Rs * Vs','fro');

for i = 1:size(xdata,2)
   x = xdata(:,i); 
   [Us, Rs, Vs, normE] = updateTURVD(Xi,Us, Rs, Vs, x, normE, epsilon);
   Xi = [Xi, x];
   fprintf('%d th iteration, err = %g, trunc = %d\n',i, norm(Xi - Us * Rs * Vs','fro'), size(Us,2));
end
