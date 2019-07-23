#include "network.h"
#include "mtcnn.hpp"
#include <time.h>
#include <stdio.h>
#include "network.h"



int main()
{
	clock_t start;
	double total_time;
	char photoName2[]="01.jpg";
	char photoName4[]="04.jpg";
	
	// test Img 4
	//string 
    Mat image = imread(photoName4);
	
	
	//4.jpg------------------
	int right_faceNum=5;
	//int rows=480,cols=640;//4.jpg
	
	//printf("image.rows=%d, image.cols=%d\n",rows, cols);
    mtcnn find(image.rows, image.cols);
	

    start = clock();
    int FaceNum=find.findFace(image);
	total_time = (double)(clock() -start)/CLOCKS_PER_SEC*1000;
    cout<<"All find time is  "<<total_time<<" milli second"<<endl;
    imshow("result", image);
    //imwrite("result.jpg",image);
	image.release();
	
	if(FaceNum==right_faceNum)
		printf("SUCCESS!\n");
	else
		printf("Program ERROR!\n");
	
	waitKey(0);
	
	//test img 2
	image = imread(photoName2);
	right_faceNum=1;
	
	mtcnn find2(image.rows, image.cols);
	start = clock();
	FaceNum=find2.findFace(image);
	
	total_time = (double)(clock() -start)/CLOCKS_PER_SEC*1000;
    cout<<"All find2 time is  "<<total_time<<" milli second"<<endl;
    imshow("result", image);
    //imwrite("result.jpg",image);
	image.release();
	
	if(FaceNum==right_faceNum)
		printf("SUCCESS!\n");
	else
		printf("Program ERROR!\n");
	
    waitKey(0);
	
	
	
    return 0;
}
