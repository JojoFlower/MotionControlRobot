#include "opencv2/opencv.hpp"
#include <complex>
#include <iostream>
#include <valarray>
#include <fftw3.h>
#include <opencv2/plot.hpp>

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
		std::cout << "Nb contour: " << contours.size() << endl;
		std::cout << "Long contour: " << contours.at(0).size() << endl;
		std::cout << "(x,y): " << contours.at(0).at(0).x << ", " << contours.at(0).at(0).y << endl;
		unsigned int max = 0;
		int j = 0;
		std:vector<Point> contour;
		for(auto i = contours.begin(); i != contours.end(); i++){
			if(i->size() > max){
				max = i->size();
				numc = j;
				contour = *i;
			};
			j++;
		};
		unsigned int contour_size = contour.size();
		unsigned int sum_x;
		unsigned int sum_y;
		fftw_complex in[contour_size], out[contour_size];	
		fftw_plan p;	
		for(unsigned int i = 0; i < contour_size; i++){
			in[i][0] = contour[i].x;
			in[i][1] = contour[i].y;
			sum_x += contour[i].x;
			sum_y += contour[i].y;
		}
		sum_x = sum_x/contour_size;
		sum_y = sum_y/contour_size;
		for(unsigned int i = 0; i < contour_size; i++){
			in[i][0] = in[i][0]/sum_x;
			in[i][1] = in[i][1]/sum_y;
		}

		p = fftw_plan_dft_1d(contour_size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(p);
		fftw_destroy_plan(p);
		fftw_cleanup();
		for(unsigned int i = 0; i < contour_size; i++){
			out[i][0] = in[i][0]/contour_size;
			in[i][1] = in[i][1]/contour_size;
		}
		// std::cout << "Contour max: " << contour.size() << endl;
		// for(int i =0; i<15; i++){
		// 	std::cout << out[i][0] << ", " << out[i][1] << endl;
		// }
		Mat xData, yData, display;
		Ptr<plot::Plot2d> plot;
		xData.create(1, 100, CV_64F);//1 Row, 100 columns, Double
		yData.create(1, 100, CV_64F);

		for(int i = 0; i<100; ++i)
		{
			xData.at<double>(i) = i/10.0;
			yData.at<double>(i) = i/10.0;
		}
		plot = plot::createPlot2d(xData, yData);
		plot->setPlotSize(100, 1000);
		plot->setMaxX(10);
		plot->setMinX(0);
		plot->setMaxY(100);
		plot->setMinY(-1);
		plot->render(display);
		imshow("Plot", display);
		waitKey();

		Mat Dessin = Mat::zeros(X,Y, CV_8UC1);
		// for(numc = 0; numc < contours.size(); numc++ )
		// 	drawContours( Dessin, contours, numc, 255);
		drawContours( Dessin, contours, numc, 255);
		imshow("Contours", Dessin);
		if(waitKey(30) != 255) break;
	}
	return 0;
}


