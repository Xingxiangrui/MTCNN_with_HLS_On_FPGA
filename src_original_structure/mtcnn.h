#ifndef MTCNN_H
#define MTCNN_H
#include "network.h"

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
	//conv1,prelu1
	struct Weight *conv1_wb;
    struct pRelu *prelu_gmma1;//weight
    struct pBox *conv1_matrix_in;//input
    struct pBox *conv1_out;//output
	//pool1
    struct pBox *maxPooling1_out;//output
	//conv2,prelu2
	struct Weight *conv2_wb;
    struct pRelu *prelu_gmma2;//weight
    struct pBox *conv2_matrix_in;//input
    struct pBox *conv2_out;//output
	//conv3,prelu3
	struct Weight *conv3_wb;
    struct pRelu *prelu_gmma3;//weight
    struct pBox *conv3_matrix_in;//input
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
	
	//conv1,prelu1
	struct Weight *conv1_wb;
    struct pRelu *prelu_gmma1;//weight
    struct pBox *conv1_matrix;//input
    struct pBox *conv1_out;//output
	//pool1
    struct pBox *pooling1_out;

	//conv2,prelu2
	struct Weight *conv2_wb;
    struct pRelu *prelu_gmma2;//weight
    struct pBox *conv2_matrix;//input
    struct pBox *conv2_out;//output
	
	//pool2
    struct pBox *pooling2_out;

	//conv3,prelu3
	struct Weight *conv3_wb;
    struct pRelu *prelu_gmma3;//weight
    struct pBox *conv3_matrix;//input
    struct pBox *conv3_out;//output

	//post conv process
	struct Weight *fc4_wb;
    struct pRelu *prelu_gmma4;//weight
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
    struct pBox *keyPoint_;
private:
    struct pBox *rgb;
	
	//conv1,prelu1
	struct Weight *conv1_wb;
    struct pRelu *prelu_gmma1;//Weight
    struct pBox *conv1_matrix;
    struct pBox *conv1_out;//output
	
	//pool1
    struct pBox *pooling1_out;

	//conv2,prelu2
	struct Weight *conv2_wb;
    struct pRelu *prelu_gmma2;//Weight
    struct pBox *conv2_matrix;//input
    struct pBox *conv2_out;//output
	//pool2
    struct pBox *pooling2_out;

	//conv3,prelu3
	struct Weight *conv3_wb;
    struct pRelu *prelu_gmma3;//Weight
    struct pBox *conv3_matrix;//input
    struct pBox *conv3_out;//output
	
	//pool3
    struct pBox *pooling3_out;

	//conv4,prelu4
	struct Weight *conv4_wb;
    struct pRelu *prelu_gmma4;//Weight
    struct pBox *conv4_matrix;//input
    struct pBox *conv4_out;//output

	//post conv precess
	struct Weight *fc5_wb;
    struct pRelu *prelu_gmma5;//Weight
	struct Weight *score_wb;
    struct Weight *location_wb;
    struct Weight *keyPoint_wb;//Weight
    struct pBox *fc5_out;//output

    void OnetImage2MatrixInit(struct pBox *pbox);
};

class mtcnn
{
public:
    mtcnn(int row, int col);
    ~mtcnn();
    void findFace(Mat &image);
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