#include "network.h"

void addbias(struct pBox *pbox, mydataFmt *pbias){
    if (pbox->pdata == NULL){
        cout << "Relu feature is NULL!!" << endl;
        return;
    }
    if (pbias == NULL){
        cout << "the  Relu bias is NULL!!" << endl;
        return;
    }
    mydataFmt *op = pbox->pdata;
    mydataFmt *pb = pbias;

    long dis = pbox->width*pbox->height;
    for(int channel =0;channel<pbox->channel; channel++){
        for(int col=0; col<dis; col++){
            *op = *op + *pb;
            op++;
        }
        pb++;
    }
}
void image2MatrixInit(Mat &image, struct pBox *pbox){
    if ((image.data == NULL) || (image.type() != CV_8UC3)){
        cout << "image's type is wrong!!Please set CV_8UC3" << endl;
        return;
    }
    pbox->channel = image.channels();
    pbox->height = image.rows;
    pbox->width = image.cols;
    
    pbox->pdata = (mydataFmt *)malloc(pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
    if(pbox->pdata==NULL)cout<<"the image2MatrixInit failed!!"<<endl;
    memset(pbox->pdata, 0, pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
}
//------------RGB and feature data order--------------------
//  for channels for rows for cols
void image2Matrix(const Mat &image, const struct pBox *pbox){
    if ((image.data == NULL) || (image.type() != CV_8UC3)){
        cout << "image's type is wrong!!Please set CV_8UC3" << endl;
        return;
    }
    if (pbox->pdata == NULL){
        return;
    }
    mydataFmt *p = pbox->pdata;
    for (int rowI = 0; rowI < image.rows; rowI++){
        for (int colK = 0; colK < image.cols; colK++){
            *p = (image.at<Vec3b>(rowI, colK)[0] - 127.5)*0.0078125;
            *(p + image.rows*image.cols) = (image.at<Vec3b>(rowI, colK)[1] - 127.5)*0.0078125;
            *(p + 2*image.rows*image.cols) = (image.at<Vec3b>(rowI, colK)[2] - 127.5)*0.0078125;
            p++;
        }
    }
	
	//write pBox format image data into the bin file 
/* 	printf("----------start write into .bin-----\n");
	char fileName[]="image1.bin";
	FILE * binFILE_ptr;
	if((binFILE_ptr=fopen(fileName,"wb"))==NULL){
		printf("can't open weight file");
		exit(0);
	}
	fwrite(pbox->pdata, sizeof(float),3*image.rows*image.cols,binFILE_ptr);
	fclose(binFILE_ptr);
	printf("----------done write  .bin-----\n"); */
	
}

//Our new pad function: left and right pad can be different
void featurePadInit(const pBox *pbox, pBox *outpBox, const int leftPad,const int rightPad){
    if ((rightPad<=0)&&(leftPad <= 0)){
        cout << "the data needn't to pad,please check you network!" << endl;
        return;
    }
    outpBox->channel = pbox->channel;
    outpBox->height = pbox->height + rightPad+leftPad;
    outpBox->width = pbox->width + rightPad+leftPad;
    long RowByteNum= outpBox->width*sizeof(mydataFmt);
    outpBox->pdata = (mydataFmt *)malloc(outpBox->channel*outpBox->height*RowByteNum);
    if(outpBox->pdata==NULL)cout<<"the featurePadInit is failed!!"<<endl;
    memset(outpBox->pdata, 0, outpBox->channel*outpBox->height*RowByteNum);
}
void featurePad(const pBox *pboxIn, pBox *outpBox, const int leftPad,const int rightPad){
    mydataFmt *p = outpBox->pdata;
    mydataFmt *pIn = pboxIn->pdata;
	//for each row in outpBox(we dont know whether left of right pad is the left or right of feature-map)
    for (int row = 0; row < outpBox->channel*outpBox->height;row++){
        if ((row%outpBox->height) <leftPad || (row % outpBox->height >(outpBox->height-rightPad-1))){
            p += outpBox->width;
            continue;//end corrent loop and to the loop judge
        }
        p += leftPad;
        memcpy(p, pIn, pboxIn->width*sizeof(mydataFmt));
        p += pboxIn->width + rightPad;
        pIn += pboxIn->width;
    }
}

// -------------convulution in 2D matrix format-------------------
// input kernel matrix  *  input feature matrix(Trans) = output feature matrix
// height (outChannels)    height (3D_KernelSize)        height (outChannels)
// width  (3D_KernelSize)  width  (outFeatureSize)       width  (outFeatureSize)
//Feature Matrix: MatrixOut->height(outFeatureSize) MatrixOut->width(3D_KernelSize)
void feature2MatrixInit(const pBox *pboxIn, pBox *MatrixOut, const Weight *weight){
//just feature2matrix not consider the pad process and input already paded
    int kernelSize = weight->kernelSize;
    int stride = weight->stride;
	int leftPad=weight->leftPad;
	int rightPad=weight->rightPad;
    int w_out = (pboxIn->width - kernelSize) / stride + 1;
    int h_out = (pboxIn->height - kernelSize) / stride + 1;
    MatrixOut->width = pboxIn->channel*kernelSize*kernelSize;//3D_KernelSize
    MatrixOut->height = w_out*h_out;//outFeatureSize
    MatrixOut->channel = 1;
    MatrixOut->pdata = (mydataFmt *)malloc(MatrixOut->width*MatrixOut->height*sizeof(mydataFmt));
    if(MatrixOut->pdata==NULL)cout<<"the feature2MatrixInit failed!!"<<endl;
    memset(MatrixOut->pdata, 0, MatrixOut->width*MatrixOut->height*sizeof(mydataFmt));
}
void feature2Matrix(const pBox *pboxIn, pBox *MatrixOut, const Weight *weight){
    if (pboxIn->pdata == NULL){
        cout << "the feature2Matrix pboxIn is NULL!!" << endl;
        return;
    }
    int kernelSize = weight->kernelSize;
    int stride = weight->stride;
	int leftPad=weight->leftPad;
	int rightPad=weight->rightPad;
    int w_out = (pboxIn->width - kernelSize) / stride + 1;
    int h_out = (pboxIn->height - kernelSize) / stride + 1;
    
    mydataFmt *p = MatrixOut->pdata;//MatrixOut
    mydataFmt *pIn;//pboxIn
    mydataFmt * ptemp;
	//from here we can get the weight is for out_row, for out_col, for channelIn
    for (int row = 0; row< h_out; row ++){//row_out
	  for (int col = 0; col < w_out; col++){//col_out
		pIn = pboxIn->pdata + row*stride*pboxIn->width + col*stride;

		for (int channel_in = 0; channel_in < pboxIn->channel; channel_in++){//channel_in
			ptemp = pIn + channel_in*pboxIn->height*pboxIn->width;
			for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++){//kernelSize
				memcpy(p, ptemp, kernelSize*sizeof(mydataFmt));//from ptemp to p
				p += kernelSize;
				ptemp += pboxIn->width;
			}
		}
	  }
    }
}

void convolutionInit(const Weight *weightIn, const pBox *pboxParameter, pBox *outpBox){
	outpBox->channel = weightIn->out_ChannelNum;
    outpBox->width = (pboxParameter->width - weightIn->kernelSize) / weightIn->stride + 1;
    outpBox->height = (pboxParameter->height - weightIn->kernelSize) / weightIn->stride + 1;
	int outpBoxByteSize=weightIn->out_ChannelNum*outpBox->width*outpBox->height*sizeof(mydataFmt);
    outpBox->pdata = (mydataFmt *)malloc(outpBoxByteSize);
    
	if(outpBox->pdata==NULL)cout<<"the convolutionInit is failed!!"<<endl;
    memset(outpBox->pdata , 0, outpBoxByteSize);
}

void convolution(const Weight *weightIn, const pBox *pboxIn, pBox *outpBox){
// -------------Old convulution in 2D matrix format-------------------
// input Weight matrix  *  input feature matrix(Trans) = output feature matrix
// height (outChannels)    height (3D_KernelSize)        height (outChannels)
// width  (3D_KernelSize)  width  (outFeatureSize)       width  (outFeatureSize)
    if (pboxIn->pdata == NULL){
        cout << "the feature is NULL!!" << endl;
        return;
    }
    if (weightIn->pdata == NULL){
        cout << "the weight is NULL!!" << endl;
        return;
    }

//---------convolution in nested for loop format------------------
	//current varable for loop
	int cur_channel_out,cur_channel_in,cur_row_out,cur_col_out;
	int filter_col,filter_row;
	//network parameters
	int stride = weightIn->stride;
	int kernelSize=weightIn->kernelSize,kernelSize_2D=weightIn->kernelSize*weightIn->kernelSize;//kernel
	int out_height=outpBox->height,out_width=outpBox->width;
	int in_height=pboxIn->height,in_width=pboxIn->width;
	int in_ChannelNum=weightIn->in_ChannelNum,out_ChannelNum=weightIn->out_ChannelNum;
	int out_featureSize=out_ChannelNum*out_height*out_width;
	//location offset varable
	int output_loc,weight_pre_loc,input_pre_loc,weight_loc,input_loc;
	//three variable pointer
	float* weight_ptr=weightIn->pdata;float *input_ptr=pboxIn->pdata;float *output_ptr=outpBox->pdata;
	float sum;
	
	#pragma omp parallel for
	//set the output value to 0
	for(cur_col_out=0;cur_col_out<out_featureSize;cur_col_out++)
		output_ptr[cur_col_out]=0;

	#pragma omp parallel for
	for(cur_channel_out=0; cur_channel_out<out_ChannelNum; cur_channel_out++){//out_channel
	 for(cur_channel_in=0;cur_channel_in<in_ChannelNum;cur_channel_in++){//in_channel
	  for(cur_row_out=0;cur_row_out<out_height;cur_row_out++){//out_row,out_height
		for(cur_col_out=0;cur_col_out<out_width;cur_col_out++){//out_col,out_width
			output_loc=cur_channel_out*out_height*out_width+cur_row_out*out_width+cur_col_out;
			weight_pre_loc=cur_channel_out*in_ChannelNum*kernelSize_2D + cur_channel_in*kernelSize_2D;
			input_pre_loc=cur_channel_in*in_width*in_height  \
									+ cur_row_out*stride*in_width+stride*cur_col_out;
			sum=0;
// outpBox [out_ChannelNum][out_height][out_width] +=
//		weightIn[out_ChannelNum][in_ChannelNum][kernelWidth][kernelHeight] *
//		pboxIn[in_ChannelNum][width][height] 
			for (filter_row=0;filter_row<kernelSize;filter_row++){
			  for(filter_col=0;filter_col<kernelSize;filter_col++){
				weight_loc=weight_pre_loc+filter_row*kernelSize+filter_col;
				input_loc=input_pre_loc+filter_row*in_width+filter_col;
				sum+=weight_ptr[weight_loc]*input_ptr[input_loc];
			  }
			}
			output_ptr[output_loc]+=sum;
		}
      }
	 }
	}
}//------------------old convolution on software-------------------------


void maxPoolingInit(const pBox *pboxIn, pBox *MatrixOut, int kernelSize, int stride){
    MatrixOut->width = ceil((float)(pboxIn->width - kernelSize) / stride + 1);
    MatrixOut->height = ceil((float)(pboxIn->height - kernelSize) / stride + 1);
    MatrixOut->channel = pboxIn->channel;
    MatrixOut->pdata = (mydataFmt *)malloc(MatrixOut->channel*MatrixOut->width*MatrixOut->height*sizeof(mydataFmt));
    if(MatrixOut->pdata==NULL)cout<<"the maxPoolingInit is failed!!"<<endl;
    memset(MatrixOut->pdata, 0, MatrixOut->channel*MatrixOut->width*MatrixOut->height*sizeof(mydataFmt));
}
void maxPooling(const pBox *pboxIn, pBox *MatrixOut, int kernelSize, int stride){
    if (pboxIn->pdata == NULL){
        cout << "the feature2Matrix pbox is NULL!!" << endl;
        return;
    }
    mydataFmt *p = MatrixOut->pdata;
    mydataFmt *pIn;
    mydataFmt *ptemp;
    mydataFmt maxNum = 0;
    if((pboxIn->width-kernelSize)%stride==0){
        for (int row = 0; row< MatrixOut->height; row ++){
            for (int col = 0; col < MatrixOut->width; col++){
                pIn = pboxIn->pdata + row*stride*pboxIn->width + col*stride;
                for (int channel = 0; channel < pboxIn->channel; channel++){
                    ptemp = pIn + channel*pboxIn->height*pboxIn->width;
                    maxNum = *ptemp;
                    for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++){
                        for(int i=0;i<kernelSize;i++){
                            if(maxNum<*(ptemp+i+kernelRow*pboxIn->width))maxNum=*(ptemp+i+kernelRow*pboxIn->width);
                        }
                    }
                    *(p+channel*MatrixOut->height*MatrixOut->width) = maxNum;
                }
                p++;
            }
        }
    }
    else{
        int diffh = 0, diffw = 0;
        for (int channel = 0; channel < pboxIn->channel; channel++){  
            pIn = pboxIn->pdata + channel*pboxIn->height*pboxIn->width;
            for (int row = 0; row< MatrixOut->height; row ++){
                for (int col = 0; col < MatrixOut->width; col++){
                    ptemp = pIn + row*stride*pboxIn->width + col*stride;
                    maxNum = *ptemp;
                    diffh = row*stride-pboxIn->height+1;
                    diffw = col*stride-pboxIn->height+1;
                    for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++){
                        if((kernelRow+diffh)>0)break;
                        for(int i=0;i<kernelSize;i++){
                            if((i+diffw)>0)break;
                            if(maxNum<*(ptemp+i+kernelRow*pboxIn->width))maxNum=*(ptemp+i+kernelRow*pboxIn->width);
                        }
                    }
                    *p++ = maxNum;
                }
            }
        }
    }
}
void relu(struct pBox *pbox, mydataFmt *pbias){
    if (pbox->pdata == NULL){
        cout << "the  Relu feature is NULL!!" << endl;
        return;
    }
    if (pbias == NULL){
        cout << "the  Relu bias is NULL!!" << endl;
        return;
    }
    mydataFmt *op = pbox->pdata;
    mydataFmt *pb = pbias;

    long dis = pbox->width*pbox->height;
    for(int channel =0;channel<pbox->channel; channel++){
        for(int col=0; col<dis; col++){
            *op += *pb;
            if(*op<0)*op=0;
            op++;
        }
        pb++;
    }
}
void prelu(struct pBox *pbox, mydataFmt *pbias, mydataFmt *prelu_gmma){
    if (pbox->pdata == NULL){
        cout << "the  Relu feature is NULL!!" << endl;
        return;
    }
    if (pbias == NULL){
        cout << "the  Relu bias is NULL!!" << endl;
        return;
    }
    mydataFmt *op = pbox->pdata;
    mydataFmt *pb = pbias;
    mydataFmt *pg = prelu_gmma;

    long dis = pbox->width*pbox->height;
    for(int channel =0;channel<pbox->channel; channel++){
        for(int col=0; col<dis; col++){
            *op = *op + *pb;
            *op = (*op>0)?(*op):((*op)*(*pg));
            op++;
        }
        pb++;
        pg++;
    }
}
void fullconnectInit(const Weight *weight, pBox *outpBox){

    outpBox->channel = weight->out_ChannelNum;
    outpBox->width = 1;
    outpBox->height = 1;
    outpBox->pdata = (mydataFmt *)malloc(weight->out_ChannelNum*sizeof(mydataFmt));
    if(outpBox->pdata==NULL)cout<<"the fullconnectInit is failed!!"<<endl;
    memset(outpBox->pdata, 0, weight->out_ChannelNum*sizeof(mydataFmt));
}

// -------------Fully connected layers in matrix*vector format-------------------
// input weight matrix  *  input feature vector =   output feature matrix
// height (outFeaSize)     height (inFeaSize)       height (outFeaSize)
// width  (inFeaSize)    
void fullconnect(const Weight *weight, const pBox *Inpbox, pBox *outpBox){
    if (Inpbox->pdata == NULL){
        cout << "the fc feature is NULL!!" << endl;
        return;
    }
    if (weight->pdata == NULL){
        cout << "the fc weight is NULL!!" << endl;
        return;
    }
    memset(outpBox->pdata, 0, weight->out_ChannelNum*sizeof(mydataFmt));
// -------------Fully connected layers in matrix*vector format--------------
// input weight matrix  *  input feature vector =   output feature matrix
// height (outFeaSize)     height (inFeaSize)       height (outFeaSize)
// width  (inFeaSize)    
//---------------------original FC calculate----------------------
/* 	//Y=αAX + βY    β must be 0(zero)  cblas_sgemv:Multiplies a matrix by a vector (single precision)
	//          row_Major      no_trans      A hight                 A width               alpha
	cblas_sgemv(CblasRowMajor, CblasNoTrans, weight->out_ChannelNum, weight->in_ChannelNum,1,   \
	//A*           A width                x               1   beta  C*              1
	weight->pdata, weight->in_ChannelNum, Inpbox->pdata,  1,  0,    outpBox->pdata, 1); */
//---------------------original FC calculate end-------------------

//C=αAB + βC :   outpBox=weightIn*matrixIn(T)
/* 	//       A_transpose       B_transpose
	gemm_cpu(0,                0,            \
	//A hight C hight       B width C width   A width B hight         
	weight->out_ChannelNum, 1,                weight->in_ChannelNum,      \
	//A*             A'width                B*              B'width   
	weight->pdata,   weight->in_ChannelNum, Inpbox->pdata,  1,         \
	//C*             C'width
	outpBox->pdata,  1); */
	
//--------------------fc layer	in nested loop format--------------
	//loop variables
	int cur_outChannel,cur_inChannel;
	int out_ChannelNum=weight->out_ChannelNum, in_ChannelNum=weight->in_ChannelNum;
	//loaction variables
	int weight_loc_pre,weight_loc;
	//variable pointer
	float sum;
	for(cur_outChannel=0;cur_outChannel<out_ChannelNum;cur_outChannel++){
		sum=0;
		weight_loc_pre=cur_outChannel*in_ChannelNum;
		for(cur_inChannel=0;cur_inChannel<in_ChannelNum;cur_inChannel++){
			weight_loc=weight_loc_pre+cur_inChannel;
			sum+=weight->pdata[weight_loc]*Inpbox->pdata[cur_inChannel];
		}
		outpBox->pdata[cur_outChannel]=sum;
	}
}

void readData(string filename, long dataNumber[], mydataFmt *pTeam[], int prtNum){
    
	FILE * weightFILE_ptr;
	if((weightFILE_ptr=fopen(filename.data(),"rb"))==NULL){
		printf("can't open weight file");
		exit(0);
	}
		
	int count = 0;
	//while(!feof(weightFILE_ptr)){
	while(count<prtNum){
		fread(pTeam[count],sizeof(mydataFmt),dataNumber[count],weightFILE_ptr);
		//printf("count=%d dataNumber[count]=%ld \n",count,dataNumber[count]);
		count++;
	}

	fclose(weightFILE_ptr);
}

long initConvAndFc(struct Weight *weight, int schannel, int lchannel, int kersize, int stride, int leftPad,int rightPad){
    weight->out_ChannelNum = schannel;
    weight->in_ChannelNum = lchannel;
    weight->kernelSize = kersize;
    weight->stride = stride;
    weight->leftPad = leftPad;
	  weight->rightPad = rightPad;
    weight->pbias = (mydataFmt *)malloc(schannel*sizeof(mydataFmt));
    if(weight->pbias==NULL)cout<<"In initConvAndFc weight buffer malloc failure!";
    memset(weight->pbias, 0, schannel*sizeof(mydataFmt));
    long byteLenght = weight->out_ChannelNum*weight->in_ChannelNum*weight->kernelSize*weight->kernelSize;
    weight->pdata = (mydataFmt *)malloc(byteLenght*sizeof(mydataFmt));
    if(weight->pdata==NULL)cout<<"In initConvAndFc weight buffer malloc failure!";
    memset(weight->pdata, 0, byteLenght*sizeof(mydataFmt));

    return byteLenght;
}
void initpRelu(struct pRelu *prelu, int width){

    prelu->width = width;
    prelu->pdata = (mydataFmt *)malloc(width*sizeof(mydataFmt));
    if(prelu->pdata==NULL)cout<<"prelu apply for memory failed!!!!";
    memset(prelu->pdata, 0, width*sizeof(mydataFmt));
}
void softmax(const struct pBox *pbox){
    if(pbox->pdata==NULL){
        cout<<"the softmax's pdata is NULL , Please check !"<<endl;
        return;
    }
    mydataFmt *p2D = pbox->pdata;
    mydataFmt *p3D = NULL;
    long mapSize = pbox->width*pbox->height;
    mydataFmt eleSum = 0;
    for(int row=0;row<pbox->height;row++){
        for(int col=0;col<pbox->width;col++){
            eleSum = 0;
            for(int channel=0;channel<pbox->channel;channel++){
                p3D = p2D + channel*mapSize;
                *p3D = exp(*p3D);
                eleSum += *p3D;
            }
            for(int channel=0;channel<pbox->channel;channel++){
                p3D = p2D + channel*mapSize;
                *p3D = (*p3D)/eleSum;
            }
            p2D++;
        }
    }
}

bool cmpScore(struct orderScore lsh, struct orderScore rsh){
    if(lsh.score<rsh.score)
        return true;
    else
        return false;
}
void nms(vector<struct Bbox> &boundingBox_, vector<struct orderScore> &bboxScore_, const float overlap_threshold, string modelname){
    printf("Run nms\n");
	if(boundingBox_.empty()){
        return;
    }
    std::vector<int> heros;
    //sort the score
    sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);

    int order = 0;
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    while(bboxScore_.size()>0){
        order = bboxScore_.back().oriOrder;
        bboxScore_.pop_back();
        if(order<0)continue;
        heros.push_back(order);
        boundingBox_.at(order).exist = false;//delete it

        for(int num=0;num<boundingBox_.size();num++){
            if(boundingBox_.at(num).exist){
                //the iou
                maxX = (boundingBox_.at(num).x1>boundingBox_.at(order).x1)?boundingBox_.at(num).x1:boundingBox_.at(order).x1;
                maxY = (boundingBox_.at(num).y1>boundingBox_.at(order).y1)?boundingBox_.at(num).y1:boundingBox_.at(order).y1;
                minX = (boundingBox_.at(num).x2<boundingBox_.at(order).x2)?boundingBox_.at(num).x2:boundingBox_.at(order).x2;
                minY = (boundingBox_.at(num).y2<boundingBox_.at(order).y2)?boundingBox_.at(num).y2:boundingBox_.at(order).y2;
                //maxX1 and maxY1 reuse 
                maxX = ((minX-maxX+1)>0)?(minX-maxX+1):0;
                maxY = ((minY-maxY+1)>0)?(minY-maxY+1):0;
                //IOU reuse for the area of two bbox
                IOU = maxX * maxY;
                if(!modelname.compare("Union"))
                    IOU = IOU/(boundingBox_.at(num).area + boundingBox_.at(order).area - IOU);
                else if(!modelname.compare("Min")){
                    IOU = IOU/((boundingBox_.at(num).area<boundingBox_.at(order).area)?boundingBox_.at(num).area:boundingBox_.at(order).area);
                }
                if(IOU>overlap_threshold){
                    boundingBox_.at(num).exist=false;
                    for(vector<orderScore>::iterator it=bboxScore_.begin(); it!=bboxScore_.end();it++){
                        if((*it).oriOrder == num) {
                            (*it).oriOrder = -1;
                            break;
                        }
                    }
                }
            }
        }
    }
    for(int i=0;i<heros.size();i++)
        boundingBox_.at(heros.at(i)).exist = true;
}
void refineAndSquareBbox(vector<struct Bbox> &vecBbox, const int &height, const int &width){
    if(vecBbox.empty()){
        cout<<"Bbox is empty!!"<<endl;
        return;
    }
    float bbw=0, bbh=0, maxSide=0;
    float h = 0, w = 0;
    float x1=0, y1=0, x2=0, y2=0;
    for(vector<struct Bbox>::iterator it=vecBbox.begin(); it!=vecBbox.end();it++){
        if((*it).exist){
            bbh = (*it).x2 - (*it).x1 + 1;
            bbw = (*it).y2 - (*it).y1 + 1;
            x1 = (*it).x1 + (*it).regreCoord[1]*bbh;
            y1 = (*it).y1 + (*it).regreCoord[0]*bbw;
            x2 = (*it).x2 + (*it).regreCoord[3]*bbh;
            y2 = (*it).y2 + (*it).regreCoord[2]*bbw;

            h = x2 - x1 + 1;
            w = y2 - y1 + 1;
          
            maxSide = (h>w)?h:w;
            x1 = x1 + h*0.5 - maxSide*0.5;
            y1 = y1 + w*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);

            //boundary check
            if((*it).x1<0)(*it).x1=0;
            if((*it).y1<0)(*it).y1=0;
            if((*it).x2>height)(*it).x2 = height - 1;
            if((*it).y2>width)(*it).y2 = width - 1;

            it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
        }
    }
}

//---------------gemm in openBLAS------------------
void gemm_cpu(int TA, int TB, int M, int N, int K,  
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
	int i,j;
	for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] = 0;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, A,lda, B, ldb,C,ldc);
}

void gemm_nn(int M, int N, int K, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K,
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}
