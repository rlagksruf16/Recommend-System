function X=geometric(m,n,kappa)
     p=min([m,n]);
     lnkappa=-log(kappa)/(p-1);
     exponents=(0:p-1)*lnkappa;
     sigma=exp(exponents);
     if m > n
        U=orth(randn(m,n));  V=orth(randn(n,n));
    else
	U=orth(randn(m,m));  V=orth(randn(n,m));
    end;
    X=U*diag(sigma)*V';
