% fonction lirepgm
% usage : I=lirepgm(nom)
% nom : nom du fichier image avec extension PGM
% I : tableau à 2 dimensions (codage des niveaux en intensité entre 0.0 et 1.0)
% I(ligne,colonne)
% lancer imshow(I) pour visualiser l'image

function I=lirepgm(nom)
fid=fopen(nom,'r');
if fid == -1
    error('fichier non ouvert');
end
[LIG,COL]=parcoursentete_pgm(fid);
[buffer,nblu]=fread(fid,LIG*COL,'uchar');
fclose(fid);
if nblu ~= LIG*COL
    error('fichier incomplet');
end
I=zeros(COL,LIG);
I(:)=buffer(:)/255;
I=I';

function [LIG,COL]=parcoursentete_pgm(fid)
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
                if strcmp(item(1:2),'P5')==0
                    error('le seul format PGM reconnu est P5');
                end
            elseif indexitem==1
                COL=sscanf(item,'%d');
            elseif indexitem==2
                LIG=sscanf(item,'%d');
            elseif indexitem==3
                valmax=sscanf(item,'%d');
                if valmax~=255
                    error('le seul format PGM reconnu est avec valmax=255');
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
