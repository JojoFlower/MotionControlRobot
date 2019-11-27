% kmoyennes.m
% [vecteursliste,code,occur]=kmoyennes(vecteurs,dico,numiter)
% vecteurs :  vecteurs de la base (rangés en colonnes)
% dico : nombre de classes
% numiter : nombre d'essais pour trouver le dictionnaire
% vecteursliste : contient les vecteurs prototypes
% code : contient les index des classes
% occur : nombre d'éléments dans chaque classe

function [vecteursliste,code,occur]=kmoyennes(vecteurs,dico,numiter)

% nombre de lignes du tableau vecteurs : dimension des vecteurs
dimvec=size(vecteurs,1);
% nombre de colonnes du tableau vecteurs : nombre de vecteurs
M=size(vecteurs,2);

% ecriture des fichiers nécessaires au fonctionnment du programme quantvec
fid=fopen('vecteurs','w');
fwrite(fid,M,'int');
fwrite(fid,dimvec,'int');
fwrite(fid,dico,'int');
fwrite(fid,numiter,'int');
fwrite(fid,vecteurs,'float');
fclose(fid);

% lancement du programme quantvec
unix('./quantvec vecteurs dict code');

% lecture des fichiers résultat
fid=fopen('dict','r');
dimvec=fread(fid,1,'int');
dico=fread(fid,1,'int');
vecteursliste=fread(fid,[dimvec dico],'float');
occur=fread(fid,dico,'int');
fclose(fid);
disp('vecteurs dans la liste')
disp([dimvec dico])

fid=fopen('code','r');
code=fread(fid,M,'int');
fclose(fid);
