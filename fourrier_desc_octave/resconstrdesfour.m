function contfil=resconstrdesfour(coeff,N,cmin,cmax)

TC=zeros(N,1);

TC(1:cmax+1)=coeff(end-cmax:end);
TC(end+cmin+1:end)=coeff(1:-cmin);

contfil=ifft(TC)*N;

contfil=[contfil;contfil(1)];