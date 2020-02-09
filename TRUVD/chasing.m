function [Q, Lnew, Z] = chasing(L, v, u)

    m = length(u);
    
    [Q, R] = qr(u);
    Q = fliplr(Q');

    [Z, R] = qr(v);
    Z = fliplr(Z');
    Lnew = Q' * L * Z;

    [Q2 R] = qr(Lnew(1:m-1,1:m-1)');
    Q3 = eye(m,m); 
    Q3(1:m-1, 1:m-1) = Q2;

    Z = Z * Q3;
    Lnew = R';
    
