function [Us, Rs, Vs, normEs] = updateTURVD(X, U, R, V, x, normE, ep)

    [m, n] = size(X);

    [u, h] = RepGS(U, x);

    Us = [U, u];
    Rs = [[R; zeros(1, size(R,2))], h];
    Vs = [V, zeros(size(V,1),1); zeros(1,size(V,2)), 1];

    [sigk1, u1, v1] = power_iter(inv(Rs));
    if abs(sigk1) > 0
            sigk1 = 1 / sigk1;
    end

    if sqrt(sigk1^2 + normE^2) > ep
        normEs = normE;
        return;
    else
        [P2, Rs, Q3] = chasing(Rs, v1, u1);
        Us = Us * P2(:,1:end-1);
        Vs = Vs * Q3(:,1:end-1);
        normEs = sqrt(sigk1^2 + normE^2);
    end
    
%% %%%%%%%%%%%%%%%%
function [sigma, u, v] = power_iter(X)
    citer = 50; 
    v = randn(size(X,2),1); 
    u = randn(size(X,1),1);
    sigk = 0;
    eps = 1.0e-4;
    for i = 1:citer
        v2 = X * u;
        sigma = norm(v2); v2 = v2 / sigma;
        u2 = X' * v2;
        u2 = u2 / norm(u2);
        u = u2;
        v = v2;
        if abs(sigk - sigma) < eps 
            break;
        end
    end