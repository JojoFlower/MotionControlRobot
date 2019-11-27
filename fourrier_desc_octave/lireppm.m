% fonction lireppm
% usage : RGB=lireppm(nom)
% nom : nom du fichier image avec extension PPM
% RGB : tableau à 3 dimensions (codage des niveaux de couleur en intensité entre 0.0 et 1.0)
% RGB(ligne,colonne,composante)
% lancer imshow(RGB) pour visualiser l'image

function RGB=lireppm(nom)
fid=fopen(nom,'r');
if fid == -1
    error('fichier non ouvert');
end
[LIG,COL]=parcoursentete_ppm(fid);
[buffer,nblu]=fread(fid,3*LIG*COL,'uchar');
fclose(fid);
if nblu ~= 3*LIG*COL
    error('fichier incomplet');
end
RGB=zeros(LIG,COL,3);
R=zeros(COL,LIG);
G=zeros(COL,LIG);
B=zeros(COL,LIG);
R(1:LIG*COL)=buffer(1:3:end)/255;
G(1:LIG*COL)=buffer(2:3:end)/255;
B(1:LIG*COL)=buffer(3:3:end)/255;
RGB(:,:,1)=R';
RGB(:,:,2)=G';
RGB(:,:,3)=B';

function [LIG,COL]=parcoursentete_ppm(fid)
MAX_ITEM=100;
entete=1;
indexcarac=1;
indexitem=0;
commentaire=0;
item=char(zeros(1,MAX_ITEM));
while entete==1
    carac=fread(fid,1,'char');
    item(indexcarac)=carac;
    indexcarac=indexcarac+1;
    if indexcarac==MAX_ITEM
        error('entete trop grande');
    end
    if carac==32 || carac==10 || carac==9
        item(indexcarac-1)=0;
        if item(1)==35
            commentaire=1;
        end
        if commentaire==0
            if indexitem==0
                if strcmp(item(1:2),'P6')==0
                    error('le seul format PPM reconnu est P6');
                end
            elseif indexitem==1
                COL=sscanf(item,'%d');
            elseif indexitem==2
                LIG=sscanf(item,'%d');
            elseif indexitem==3
                valmax=sscanf(item,'%d');
                if valmax~=255
                    error('le seul format PPM reconnu est avec valmax=255');
                end
                entete=0;
            else
            end
            indexitem=indexitem+1;
        end
        if carac==10
            commentaire=0;
        end
        indexcarac=1;
    end
end
