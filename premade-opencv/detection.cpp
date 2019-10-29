#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int, char**)
{
	int seuil=30;
	char detect;
	VideoCapture cap(0); 
	if(!cap.isOpened())  
		return -1;
	namedWindow("Image",1);
	namedWindow("Detection",1);
	namedWindow("Contours",1);
	while(true)
	{
		int X,Y,DIM,index,indexNB;
		unsigned int numc;
		uchar R,G,B;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		Mat frame;
		cap >> frame;
		X=frame.rows;
		Y=frame.cols;
		Mat Binaire(X,Y,CV_8UC1);
		imshow("Image", frame);
		GaussianBlur(frame, frame, Size(7,7), 1.5, 1.5);
		X=frame.rows;
		Y=frame.cols;
		DIM=frame.channels();
		for (index=0,indexNB=0;index<DIM*X*Y;index+=DIM,indexNB++)
		{
			detect=0;
			B=frame.data[index    ];
			G=frame.data[index + 1];
			R=frame.data[index + 2];
			if ((R>G) && (R>B))
				if (((R-B)>=seuil) || ((R-G)>=seuil))
					detect=1;
			if (detect==1)
				Binaire.data[indexNB]=255;
			else
				Binaire.data[indexNB]=0;
		}
		imshow("Detection", Binaire);
		findContours( Binaire, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		Mat Dessin = Mat::zeros(X,Y, CV_8UC1);
		for(numc = 0; numc < contours.size(); numc++ )
			drawContours( Dessin, contours, numc, 255);
		imshow("Contours", Dessin);
		if(waitKey(30) != 255) break;
	}
	return 0;
}


