#ifndef MTCNN_H
#define MTCNN_H
#include "network.h"

namespace testNamespace {
	extern int testNum;
}


class Pnet
{
public:
    Pnet();
    ~Pnet();
    void run(Mat &image, float scale);

    float nms_threshold;
    mydataFmt Pthreshold;
    bool firstFlag;
    vector<struct Bbox> boundingBox_;
    vector<orderScore> bboxScore_;
private:
    //in order of [weight] [input] [output]
    //the RGB image in matrix format
    struct pBox *rgb;
    //conv1,relu1
    struct Weight *conv1_wb;
    struct pBox *conv1_out;//output
    struct pBox *conv1_out_pad;//add padding
    //pool1
    struct Weight *pool1_conv1_wb;
    struct pBox *pool1_conv1_out;//output
    //conv2,relu2
    struct Weight *conv2_wb;
    struct pBox *conv2_out;//output
    //conv3,relu3
    struct Weight *conv3_wb;
    struct pBox *conv3_out;//output
    //post conv layers
    struct Weight *conv4c1_wb;
    struct Weight *conv4c2_wb;//weight
    struct pBox *score_matrix;
    //the 4th layer's out   out
    struct pBox *score_;
    //the 4th layer's out   out
    struct pBox *location_matrix;
    struct pBox *location_;



//	//in order of [weight] [input] [output]
//    //the RGB image in matrix format
//    struct pBox *rgb;
//	//conv1,prelu1
//	struct Weight *conv1_wb;
//    struct pRelu *prelu_gmma1;//weight
//    struct pBox *conv1_matrix_in;//input
//    struct pBox *conv1_out;//output
//	//pool1
//    struct pBox *maxPooling1_out;//output
//	//conv2,prelu2
//	struct Weight *conv2_wb;
//    struct pRelu *prelu_gmma2;//weight
//    struct pBox *conv2_matrix_in;//input
//    struct pBox *conv2_out;//output
//	//conv3,prelu3
//	struct Weight *conv3_wb;
//    struct pRelu *prelu_gmma3;//weight
//    struct pBox *conv3_matrix_in;//input
//    struct pBox *conv3_out;//output
//	
//	//post conv layers
//	struct Weight *conv4c1_wb;
//    struct Weight *conv4c2_wb;//weight
//    struct pBox *score_matrix;
//    //the 4th layer's out   out
//    struct pBox *score_;
//    //the 4th layer's out   out
//    struct pBox *location_matrix;
//    struct pBox *location_;

    void generateBbox(const struct pBox *score, const struct pBox *location, mydataFmt scale);
};

class Rnet
{
public:
    Rnet();
    ~Rnet();
    float Rthreshold;
    void run(Mat &image);
    struct pBox *score_;
    struct pBox *location_;
private:
    struct pBox *rgb;	
    struct pBox *rgb_pad;
    //conv1
    struct Weight *conv1_wb;
    struct pBox *conv1_out;//output
    struct pBox *conv1_out_pad;//output
    //pool1
    struct Weight *pool_conv1_wb;
    struct pBox *pool_conv1_out;
    struct pBox *pool_conv1_out_pad;
    //conv2
    struct Weight *conv2_wb;
    struct pBox *conv2_out;//output
    struct pBox *conv2_out_pad;//output
    //pool2
    struct Weight *pool2_conv3_wb;
    struct pBox *pool2_conv3_out;//output
    struct pBox *pool2_conv3_out_pad;//output
    //conv3
    struct Weight *conv3_wb;
    struct pBox *conv3_out;//output
    struct pBox *conv3_out_pad;//output

    //post conv process
    struct Weight *fc4_wb;
    struct Weight *score_wb;
    struct Weight *location_wb;//weight
    struct pBox *fc4_out;//output

    void RnetImage2MatrixInit(struct pBox *pbox);
};

class Onet
{
public:
    Onet();
    ~Onet();
    void run(Mat &image);
    float Othreshold;
    struct pBox *score_;
    struct pBox *location_;
//    struct pBox *keyPoint_;
private:
    struct pBox *rgb;
    struct pBox *rgb_pad;
	
	  //conv1,relu1
	  struct Weight *conv1_wb;
    struct pBox *conv1_out;//output
    struct pBox *conv1_out_pad;//output

	  //conv2,relu2
	  struct Weight *conv2_wb;
    struct pBox *conv2_out;//output
    struct pBox *conv2_out_pad;//output

	  //conv3,pelu3
	  struct Weight *conv3_wb;
    struct pBox *conv3_out;//output
    struct pBox *conv3_out_pad;//output
	
	  //conv4_,pelu4
	  struct Weight *conv4_wb;
    struct pBox *conv4_out;//output
    struct pBox *conv4_out_pad;//output
    
    	//conv5_,pelu5
	  struct Weight *conv5_wb;
    struct pBox *conv5_out;//output
    struct pBox *conv5_out_pad;//output
    
    //conv6_,pelu6
	  struct Weight *conv6_wb;
    struct pBox *conv6_out;//output
    struct pBox *conv6_out_pad;//output

	  //post conv precess
	  struct Weight *fc5_wb;      
    struct pBox *fc5_out;//output      
	  struct Weight *score_wb;
    struct Weight *location_wb;
//    struct Weight *keyPoint_wb;//Weight

    void OnetImage2MatrixInit(struct pBox *pbox);
};

class mtcnn
{
public:
    mtcnn(int row, int col);
    ~mtcnn();
    int findFace(Mat &image);
private:
    Mat reImage;
    float nms_threshold[3];
    vector<float> scales_;
    Pnet *simpleFace_;
    vector<struct Bbox> firstBbox_;
    vector<struct orderScore> firstOrderScore_;
    Rnet refineNet;
    vector<struct Bbox> secondBbox_;
    vector<struct orderScore> secondBboxScore_;
    Onet outNet;
    vector<struct Bbox> thirdBbox_;
    vector<struct orderScore> thirdBboxScore_;
};



#endif