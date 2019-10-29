#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int Traitement(VideoCapture cap,int seuil,Vec3b couleur)
{
	Mat trame,gris,flou,contx,conty,cont,contbin;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int X,Y,x,y,k,nbcont,numc,index;
	cap>>trame;
	X=trame.rows;
	Y=trame.cols;
	namedWindow("Image",1);
	imshow("Image", trame);
	cvtColor(trame,gris,COLOR_BGR2GRAY);
	GaussianBlur(gris,flou,Size(5,5),0,0);
	Sobel(flou,contx,CV_64F,1,0);
	Sobel(flou,conty,CV_64F,0,1);
	cont=abs(contx)+abs(conty);
	contbin=(cont<seuil); 
	namedWindow("Gradient",1);
	imshow("Gradient",cont/255);
	findContours(contbin,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_NONE);
	Mat marqueurs = Mat::zeros(X,Y, CV_32S);
	nbcont=(int)contours.size();
	index=1;
	for(numc = 0; numc < nbcont; numc++ )
		if (hierarchy[numc][3]<0)
			drawContours( marqueurs, contours, numc, index++);
	watershed(trame,marqueurs);
	vector<double[3]> couleurs;
	vector<double> indexcoul;
	couleurs.reserve(nbcont);
	indexcoul.reserve(nbcont);
	for(index=0;index<nbcont;index++)
	{
		for(k=0;k<3;k++)
			couleurs[index][k]=0.0;
		indexcoul[index]=0.0;
	}
	for(x=0;x<X;x++)
		for(y=0;y<Y;y++)
		{
			index=marqueurs.at<int>(x,y)-1;
			if (index>=0)
			{
				indexcoul[index]++;
				for (k=0;k<3;k++)
					couleurs[index][k]=
					couleurs[index][k]+trame.at<Vec3b>(x,y)[k];
			}                    
		}
	for(index=0;index<nbcont;index++)
		for (k=0;k<3;k++)
			couleurs[index][k]/=indexcoul[index];
	for(x=0;x<X;x++)
		for(y=0;y<Y;y++)
		{
			index=marqueurs.at<int>(x,y)-1;
			if (index>=0)
				for (k=0;k<3;k++)
					trame.at<Vec3b>(x,y)[k]=couleurs[index][k];
			else
				trame.at<Vec3b>(x,y)=couleur;
		}
	namedWindow("LPE",1);
	imshow("LPE", trame);
	if(waitKey(30) >= 0)
		return true;
	else
		return false;
}

int main(int, char**)
{
	VideoCapture cap(0);
	int seuil=10;
	Vec3b couleur(128,128,128);
	while(Traitement(cap,seuil,couleur)==false);
}



