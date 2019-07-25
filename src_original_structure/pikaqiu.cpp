#include "network.h"
#include "mtcnn.h"
#include <time.h>

int main()
{
    Mat image = imread("4.jpg");
    mtcnn find(image.rows, image.cols);
    clock_t start;
	double total_time;
    start = clock();
    find.findFace(image);
    imshow("result", image);
    imwrite("result.jpg",image);
    total_time = (double)(clock() -start)/CLOCKS_PER_SEC*1000;
    cout<<"time is  "<<total_time<<" milli second"<<endl;
    // Mat image;
    // VideoCapture cap(0);
    // if(!cap.isOpened())  
    //     cout<<"fail to open!"<<endl; 
    // cap>>image;
    // if(!image.data){
    //     cout<<"读取视频失败"<<endl;  
    //     return -1;
    // }

    // mtcnn find(image.rows, image.cols);
    // clock_t start;
    // int stop = 1200;
    // while(stop--){
    //     start = clock();
    //     cap>>image;
    //     find.findFace(image);
    //     imshow("result", image);
    //     if( waitKey(1)>=0 ) break;
    //     start = clock() -start;
    //     cout<<"time is  "<<start/10e3<<endl;
    // }

    waitKey(0);
    image.release();
    return 0;
}
