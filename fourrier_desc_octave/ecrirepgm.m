% fonction ecrirepgm
% usage : ecrirepgm(nom,I)
% nom : nom du fichier image avec extension PGM
% I : tableau à 2 dimensions (codage des niveaux en intensité entre 0.0 et 1.0)
% I(ligne,colonne)
% lancer imshow(I) pour visualiser l'image

function ecrirepgm(nom,I)
LIG=size(I,1);
COL=size(I,2);
fid=fopen(nom,'w');
fprintf(fid,'P5\n%d %d\n255\n',COL,LIG);
I=I'*255;
fwrite(fid,I,'uchar');
fclose(fid);


