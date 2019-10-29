#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
   VideoCapture cap(0); 


    if ( !cap.isOpened() )  
    {
         cout << "Cannot open webcam" << endl;
         return -1;
    }
    namedWindow("MyVideo",CV_WINDOW_AUTOSIZE);
    while(1)
    {
        Mat frame;
        bool bSuccess = cap.read(frame); 
	if (!bSuccess)
	{
		cout << "Cannot read the frame from webcam" << endl;
		break;
	}

        imshow("MyVideo", frame); 

        if(waitKey(30) == 27) 
	{
                cout << "esc key is pressed by user" << endl;
		break; 
	}
    }
    return 0;
}
