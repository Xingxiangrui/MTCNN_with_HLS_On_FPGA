//c++ network  author : liqi
//Nangjing University of Posts and Telecommunications
//date 2017.5.21,20:27
#ifndef NETWORK_H
#define NETWORK_H
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <fstream>
#include <cstring>
//#include <cblas.h>
#include <string>
#include <math.h>
#include "pBox.h"
using namespace cv;

void addbias(struct pBox *pbox, mydataFmt *pbias);
void image2Matrix(const Mat &image, const struct pBox *pbox);
//void featurePad(const pBox *pbox, const pBox *outpBox, const int pad);
void featurePad(const pBox *pbox, pBox *outpBox, const int leftPad,const int rightPad);
void feature2Matrix(const pBox *pbox, pBox *Matrix, const Weight *weight);
void maxPooling(const pBox *pbox, pBox *Matrix, int kernelSize, int stride);
void relu(struct pBox *pbox, mydataFmt *pbias);
void prelu(struct pBox *pbox, mydataFmt *pbias, mydataFmt *prelu_gmma);
void convolution(const Weight *weight, const pBox *pbox, pBox *outpBox, const struct pBox *matrix);
void fullconnect(const Weight *weight, const pBox *pbox, pBox *outpBox);
void readData(string filename, long dataNumber[], mydataFmt *pTeam[],int prtNum);
//long initConvAndFc(struct Weight *weight, int schannel, int lchannel, int kersize, int stride, int pad);
long initConvAndFc(struct Weight *weight, int schannel, int lchannel, int kersize, int stride, int leftPad,int rightPad);
void initpRelu(struct pRelu *prelu, int width);
void softmax(const struct pBox *pbox);

void image2MatrixInit(Mat &image, struct pBox *pbox);
//void featurePadInit(const pBox *pbox, pBox *outpBox, const int pad);
void featurePadInit(const pBox *pbox, pBox *outpBox, const int leftPad,const int rightPad);
void maxPoolingInit(const pBox *pbox, pBox *Matrix, int kernelSize, int stride);
void feature2MatrixInit(const pBox *pbox, pBox *Matrix, const Weight *weight);
void convolutionInit(const Weight *weight, const pBox *pbox, pBox *outpBox, const struct pBox *matrix);
void fullconnectInit(const Weight *weight, pBox *outpBox);


bool cmpScore(struct orderScore lsh, struct orderScore rsh);
void nms(vector<struct Bbox> &boundingBox_, vector<struct orderScore> &bboxScore_, const float overlap_threshold, string modelname = "Union");
void refineAndSquareBbox(vector<struct Bbox> &vecBbox, const int &height, const int &width);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);	
void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
	






#endif