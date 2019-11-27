% testdesfour.m

function testdesfournorm

close all

N=200;
Nangles=500;
dico=6;
numiter=20;
cmin=-10;cmax=10;
listeangles=180*(1-2*rand(Nangles,1));

ima1=double(lirepgm('obj1.pgm')<0.5);
ima2=double(lirepgm('obj2.pgm')<0.5);
ima3=double(lirepgm('obj3.pgm')<0.5);
figure
imagesc(ima1)
title('objet 1')
axis equal
drawnow
figure
imagesc(ima2)
title('objet 2')
axis equal
drawnow
figure
imagesc(ima3)
title('objet 3')
axis equal
drawnow

tabcoeff=zeros(cmax-cmin+1,length(listeangles));
vecteurs=zeros(2*(cmax-cmin+1),length(listeangles));
tabcontfil=zeros(N+1,length(listeangles));
for n=1:length(listeangles)
    % test de l'immunité à la rotation
    choix=rand(1);
    if choix<(1/3)
        ima=ima1;
    elseif choix>(2/3)
        ima=ima2;
    else
        ima=ima3;
    end
    imar=imrotate(ima,listeangles(n),'nearest','crop');
    [coeff,ncoeff]=desfournorm(imar,cmin,cmax);
    tabcoeff(:,n)=coeff;
    contfil=resconstrdesfour(coeff,N,cmin,cmax);
    tabcontfil(:,n)=contfil;
    vecteurs(:,n)=[real(coeff);imag(coeff)];
%     plot(real(contfil),imag(contfil),'-',real(contfil(1)),imag(contfil(1)),'o')
%     grid on
%     axis equal
%     axis ij
%     drawnow
end    
figure
plot(real(tabcontfil),imag(tabcontfil),'-',real(tabcontfil(1,:)),imag(tabcontfil(1,:)),'o')
title('contours associées aux coefficients normalisés')
grid on
axis equal
drawnow
figure
plot(ncoeff,abs(tabcoeff))
title('module des coefficients')
xlabel('k');
ylabel('d_k');
grid on
zoom on
drawnow
figure
plot(vecteurs)
title('partie réellle et imaginaire des coefficients normalisés')
set(gca,'XTick',1:size(vecteurs,1))
set(gca,'XTickLabel',[ncoeff ncoeff])
grid on
ech=axis;
axis([1 size(vecteurs,1) ech(3:4)])
drawnow


vecteursliste=kmoyennes(vecteurs,dico,numiter);
 for n=1:dico
    contfil=resconstrdesfour(vecteursliste(1:end/2,n)+1i*vecteursliste(end/2+1:end,n),N,cmin,cmax);
    figure
    h=plot(real(contfil),imag(contfil),'-',real(contfil(1)),imag(contfil(1)),'o');
    title(['contour associé au vecteur prototype numéro ' int2str(n)])
    set(h(1),'LineWidth',2)
    set(h(2),'LineWidth',3)
    grid on
    axis equal
    axis ij
    drawnow
end
