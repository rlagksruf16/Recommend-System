# power Iteration matlab 코드
# function [sigma, u, v] = power_iter(X)
#     citer = 50; 
#     //무한대로 곱해야하는데 citer 50번만 곱한다
#     v = randn(size(X,2),1); 
#     u = randn(size(X,1),1);
#     sigk = 0;
#     eps = 1.0e-4;
#     for i = 1:citer
#         v2 = X * u;
#         sigma = norm(v2); v2 = v2 / sigma;
#         u2 = X' * v2;
#         u2 = u2 / norm(u2);
#         u = u2;
#         v = v2;
#         if abs(sigk - sigma) < eps 
#             break;
#         end
#     end