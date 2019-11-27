% fonction ecrireppm
% usage : ecrireppm(nom,RGB)
% nom : nom du fichier image avec extension PPM
% RGB : tableau à 3 dimensions (codage des niveaux de couleur en intensité entre 0.0 et 1.0)
% RGB(ligne,colonne,composante)
% lancer imshow(RGB) pour visualiser l'image

function ecrireppm(nom,RGB)
LIG=size(RGB,1);
COL=size(RGB,2);
fid=fopen(nom,'w');
fprintf(fid,'P6\n%d %d\n255\n',COL,LIG);
R=255*RGB(:,:,1);
G=255*RGB(:,:,2);
B=255*RGB(:,:,3);
R=R';
G=G';
B=B';
buffer=zeros(1,3*LIG*COL);
buffer(1:3:end)=R(:);
buffer(2:3:end)=G(:);
buffer(3:3:end)=B(:);
fwrite(fid,buffer,'uchar');
fclose(fid);


