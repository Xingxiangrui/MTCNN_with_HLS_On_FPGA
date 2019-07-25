#include "mtcnn.h"

Pnet::Pnet(){
    Pthreshold = 0.6;
    nms_threshold = 0.5;
    firstFlag = true;
	//in order of [weight] [input] [output]
	////the RGB image in pBox format 
    this->rgb = new pBox;

	//conv1,prelu1
	this->conv1_wb = new Weight;
    this->prelu_gmma1 = new pRelu;//weight
    this->conv1_matrix_in = new pBox;//input
    this->conv1_out = new pBox;//output
	//pool1
    this->maxPooling1_out = new pBox;
	//conv2,prelu2
	this->conv2_wb = new Weight;
    this->prelu_gmma2 = new pRelu;//weight
    this->conv2_matrix_in = new pBox;//input
    this->conv2_out = new pBox;//output
	//conv3,prelu3
	this->conv3_wb = new Weight;
    this->prelu_gmma3 = new pRelu;//weight
    this->conv3_matrix_in = new pBox;//input
    this->conv3_out = new pBox;//output
	//post conv layers
	this->conv4c1_wb = new Weight;
    this->conv4c2_wb = new Weight;//weight
    this->score_matrix = new pBox;
    this->score_ = new pBox;
    this->location_matrix = new pBox;
    this->location_ = new pBox;
    
    //weight pointer create,network initialize
    //                                 w           oc  ic  ks s lp  rp 
    long conv1 = initConvAndFc(   this->conv1_wb,  10, 3,  3, 1, 0, 0);
    initpRelu(this->prelu_gmma1, 10);
    long conv2 = initConvAndFc(   this->conv2_wb,  16, 10, 3, 1, 0, 0);
    initpRelu(this->prelu_gmma2, 16);
    long conv3 = initConvAndFc(   this->conv3_wb,  32, 16, 3, 1, 0, 0);
    initpRelu(this->prelu_gmma3, 32);
    long conv4c1 = initConvAndFc(this->conv4c1_wb, 2,  32, 1, 1, 0, 0);
    long conv4c2 = initConvAndFc(this->conv4c2_wb, 4,  32, 1, 1, 0, 0);
    long dataNumber[13] = {conv1,10,10, conv2,16,16, conv3,32,32, conv4c1,2, conv4c2,4};
    mydataFmt *pointTeam[13] = {this->conv1_wb->pdata, this->conv1_wb->pbias, this->prelu_gmma1->pdata, \
                            this->conv2_wb->pdata, this->conv2_wb->pbias, this->prelu_gmma2->pdata, \
                            this->conv3_wb->pdata, this->conv3_wb->pbias, this->prelu_gmma3->pdata, \
                            this->conv4c1_wb->pdata, this->conv4c1_wb->pbias, \
                            this->conv4c2_wb->pdata, this->conv4c2_wb->pbias \
                            };
    string filename = "Pnet.bin";
	printf("Create Pnet,Read Pnet.bin ,13 pointers \n");
    readData(filename, dataNumber, pointTeam, 13);
}
Pnet::~Pnet(){
	//in order of [weight] [input] [output]
	//input RGB image
	freepBox(this->rgb);
	//conv1,prelu1
	freeWeight(this->conv1_wb);
    freepRelu(this->prelu_gmma1);//weight
    freepBox(this->conv1_matrix_in);//input
    freepBox(this->conv1_out);//output
	//pool1
    freepBox(this->maxPooling1_out);
	//conv2,prelu2
	freeWeight(this->conv2_wb);
	freepRelu(this->prelu_gmma2);//weight
	freepBox(this->conv2_matrix_in);//input
    freepBox(this->conv2_out);//output
	//conv3,prelu3
	freeWeight(this->conv3_wb);
	freepRelu(this->prelu_gmma3);//weight
	freepBox(this->conv3_matrix_in);//input
    freepBox(this->conv3_out);//output
    
    //post conv layers
    freeWeight(this->conv4c1_wb);
	freeWeight(this->conv4c2_wb);//weight
    freepBox(this->score_);
    freepBox(this->location_);
    freepBox(this->score_matrix);
    freepBox(this->location_matrix);
    
	printf("Free Pnet\n");
}

void Pnet::run(Mat &image, float scale){
	printf("Start run Pnet\n");
    if(firstFlag){
		printf("Pnet buffer init\n");
		//change Mat image to pBox format
        image2MatrixInit(image, this->rgb);

		//conv1,prelu1
        feature2MatrixInit(this->rgb, this->conv1_matrix_in, this->conv1_wb);
        convolutionInit(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix_in);

		//pool1
        maxPoolingInit(this->conv1_out, this->maxPooling1_out, 2, 2);
		
		//conv2,prelu2
        feature2MatrixInit(this->maxPooling1_out, this->conv2_matrix_in, this->conv2_wb);
        convolutionInit(this->conv2_wb, this->maxPooling1_out, this->conv2_out, this->conv2_matrix_in);
        
		//conv3,prelu3
        feature2MatrixInit(this->conv2_out, this->conv3_matrix_in, this->conv3_wb);
        convolutionInit(this->conv3_wb, this->conv2_out, this->conv3_out, this->conv3_matrix_in);

		//post conv layers
        feature2MatrixInit(this->conv3_out, this->score_matrix, this->conv4c1_wb);
        convolutionInit(this->conv4c1_wb, this->conv3_out, this->score_, this->score_matrix);
        feature2MatrixInit(this->conv3_out, this->location_matrix, this->conv4c2_wb);
        convolutionInit(this->conv4c2_wb, this->conv3_out, this->location_, this->location_matrix);
        firstFlag = false;
    }

	//change Mat image to pBox format
    image2Matrix(image, this->rgb);

	//conv1,prelu1
    feature2Matrix(this->rgb, this->conv1_matrix_in, this->conv1_wb);
    convolution(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix_in);
    prelu(this->conv1_out, this->conv1_wb->pbias, this->prelu_gmma1->pdata);
    //pool1
    maxPooling(this->conv1_out, this->maxPooling1_out, 2, 2);

	//conv2,prelu2
    feature2Matrix(this->maxPooling1_out, this->conv2_matrix_in, this->conv2_wb);
    convolution(this->conv2_wb, this->maxPooling1_out, this->conv2_out, this->conv2_matrix_in);
    prelu(this->conv2_out, this->conv2_wb->pbias, this->prelu_gmma2->pdata);
    //conv3,prelu3
    feature2Matrix(this->conv2_out, this->conv3_matrix_in, this->conv3_wb);
    convolution(this->conv3_wb, this->conv2_out, this->conv3_out, this->conv3_matrix_in);
    prelu(this->conv3_out, this->conv3_wb->pbias, this->prelu_gmma3->pdata);
	
	//post conv layers
    //conv4c1   score
    feature2Matrix(this->conv3_out, this->score_matrix, this->conv4c1_wb);
    convolution(this->conv4c1_wb, this->conv3_out, this->score_, this->score_matrix);
    addbias(this->score_, this->conv4c1_wb->pbias);
    softmax(this->score_);
    // pBoxShow(this->score_);
    //conv4c2   location
    feature2Matrix(this->conv3_out, this->location_matrix, this->conv4c2_wb);
    convolution(this->conv4c2_wb, this->conv3_out, this->location_, this->location_matrix);
    addbias(this->location_, this->conv4c2_wb->pbias);
    //softmax layer
    generateBbox(this->score_, this->location_, scale);
	printf("Done run Pnet\n");
}
void Pnet::generateBbox(const struct pBox *score, const struct pBox *location, mydataFmt scale){
    printf("Start Pnet generate Bbox\n");
	//for pooling 
    int stride = 2;
    int cellsize = 12;
    int count = 0;
    //score p
    mydataFmt *p = score->pdata + score->width*score->height;
    mydataFmt *plocal = location->pdata;
    struct Bbox bbox;
    struct orderScore order;
    for(int row=0;row<score->height;row++){
        for(int col=0;col<score->width;col++){
            if(*p>Pthreshold){
                bbox.score = *p;
                order.score = *p;
                order.oriOrder = count;
                bbox.x1 = round((stride*row+1)/scale);
                bbox.y1 = round((stride*col+1)/scale);
                bbox.x2 = round((stride*row+1+cellsize)/scale);
                bbox.y2 = round((stride*col+1+cellsize)/scale);
                bbox.exist = true;
                bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
                for(int channel=0;channel<4;channel++)
                    bbox.regreCoord[channel]=*(plocal+channel*location->width*location->height);
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;
            }
            p++;
            plocal++;
        }
    }
	printf("Done Pnet generate Bbox\n");
}

Rnet::Rnet(){
    Rthreshold = 0.7;

    this->rgb = new pBox;
	//conv1,prelu1
    this->conv1_matrix = new pBox;
    this->conv1_out = new pBox;
	//pool1
    this->pooling1_out = new pBox;
	//conv2,prelu2
    this->conv2_matrix = new pBox;
    this->conv2_out = new pBox;
	//pool2
    this->pooling2_out = new pBox;

	//conv3,prelu3
    this->conv3_matrix = new pBox;
    this->conv3_out = new pBox;

	//post conv process
    this->fc4_out = new pBox;
    this->score_ = new pBox;
    this->location_ = new pBox;

	//weight
    this->conv1_wb = new Weight;
    this->prelu_gmma1 = new pRelu;//weight
    this->conv2_wb = new Weight;
    this->prelu_gmma2 = new pRelu;//weight
    this->conv3_wb = new Weight;
    this->prelu_gmma3 = new pRelu;//weight
    this->fc4_wb = new Weight;
    this->prelu_gmma4 = new pRelu;//weight
    this->score_wb = new Weight;
    this->location_wb = new Weight;//weight
    // //                                w         sc   lc   ks s  lp  rp
    long conv1 = initConvAndFc(   this->conv1_wb,   28, 3,   3, 1, 0,  0);
    initpRelu(this->prelu_gmma1, 28);
    long conv2 = initConvAndFc(   this->conv2_wb,   48, 28,  3, 1, 0,  0);
    initpRelu(this->prelu_gmma2, 48);
    long conv3 = initConvAndFc(   this->conv3_wb,   64, 48,  2, 1, 0,  0);
    initpRelu(this->prelu_gmma3, 64);
    long fc4 = initConvAndFc(     this->fc4_wb,     128,576, 1, 1, 0,  0);
    initpRelu(this->prelu_gmma4, 128);
    long score = initConvAndFc(   this->score_wb,   2,  128, 1, 1, 0,  0);
    long location = initConvAndFc(this->location_wb,4,  128, 1, 1, 0,  0);
    long dataNumber[16] = {conv1,28,28, conv2,48,48, conv3,64,64, fc4,128,128, score,2, location,4};
    mydataFmt *pointTeam[16] = {this->conv1_wb->pdata, this->conv1_wb->pbias, this->prelu_gmma1->pdata, \
                                this->conv2_wb->pdata, this->conv2_wb->pbias, this->prelu_gmma2->pdata, \
                                this->conv3_wb->pdata, this->conv3_wb->pbias, this->prelu_gmma3->pdata, \
                                this->fc4_wb->pdata, this->fc4_wb->pbias, this->prelu_gmma4->pdata, \
                                this->score_wb->pdata, this->score_wb->pbias, \
                                this->location_wb->pdata, this->location_wb->pbias \
                                };
    string filename = "Rnet.bin";
	printf("Create Rnet,Read Rnet.bin ,16 pointers \n");
    readData(filename, dataNumber, pointTeam, 16);

    //Init the network
	printf("Rnet buffer init\n");
	//input image to pBox format
    RnetImage2MatrixInit(rgb);
	//conv1,prelu1
    feature2MatrixInit(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolutionInit(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);
    //pool1
	maxPoolingInit(this->conv1_out, this->pooling1_out, 3, 2);
	//conv2,prelu2
    feature2MatrixInit(this->pooling1_out, this->conv2_matrix, this->conv2_wb);
    convolutionInit(this->conv2_wb, this->pooling1_out, this->conv2_out, this->conv2_matrix);
    //pool2
	maxPoolingInit(this->conv2_out, this->pooling2_out, 3, 2);
    //conv3,prelu3
	feature2MatrixInit(this->pooling2_out, this->conv3_matrix, this->conv3_wb);
    convolutionInit(this->conv3_wb, this->pooling2_out, this->conv3_out, this->conv3_matrix);
    //post conv precess
	fullconnectInit(this->fc4_wb, this->fc4_out);
    fullconnectInit(this->score_wb, this->score_);
    fullconnectInit(this->location_wb, this->location_);
}
Rnet::~Rnet(){
	printf("Free Rnet\n");
    freepBox(this->rgb);
	//conv1,prelu1
    freepBox(this->conv1_matrix);
    freepBox(this->conv1_out);
	//pool1
    freepBox(this->pooling1_out);
	//conv2,prelu2
    freepBox(this->conv2_matrix);
    freepBox(this->conv2_out);
	//pool2
    freepBox(this->pooling2_out);
	//conv3,prelu3
    freepBox(this->conv3_matrix);
    freepBox(this->conv3_out);
	
	//post conv process
    freepBox(this->fc4_out);
    freepBox(this->score_);
    freepBox(this->location_);

	//weight
    freeWeight(this->conv1_wb);
    freepRelu(this->prelu_gmma1);//weight
    freeWeight(this->conv2_wb);
    freepRelu(this->prelu_gmma2);//weight
    freeWeight(this->conv3_wb);
    freepRelu(this->prelu_gmma3);//weight
    freeWeight(this->fc4_wb);
    freepRelu(this->prelu_gmma4);//weight
    freeWeight(this->score_wb);
    freeWeight(this->location_wb);//weight
}
void Rnet::RnetImage2MatrixInit(struct pBox *pbox){
    pbox->channel = 3;
    pbox->height = 24;
    pbox->width = 24;
    
    pbox->pdata = (mydataFmt *)malloc(pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
    if(pbox->pdata==NULL)cout<<"the image2MatrixInit is failed!!"<<endl;
    memset(pbox->pdata, 0, pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
}
void Rnet::run(Mat &image){
	printf("Rnet run\n");
	//change image to pBox format
    image2Matrix(image, this->rgb);

	//conv1,prelu1
    feature2Matrix(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolution(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);
    prelu(this->conv1_out, this->conv1_wb->pbias, this->prelu_gmma1->pdata);

	//pool1
    maxPooling(this->conv1_out, this->pooling1_out, 3, 2);

	//conv2,prelu2
    feature2Matrix(this->pooling1_out, this->conv2_matrix, this->conv2_wb);
    convolution(this->conv2_wb, this->pooling1_out, this->conv2_out, this->conv2_matrix);
    prelu(this->conv2_out, this->conv2_wb->pbias, this->prelu_gmma2->pdata);
	
	//pool2
    maxPooling(this->conv2_out, this->pooling2_out, 3, 2);

    //conv3 ,prelu3
    feature2Matrix(this->pooling2_out, this->conv3_matrix, this->conv3_wb);
    convolution(this->conv3_wb, this->pooling2_out, this->conv3_out, this->conv3_matrix);
    prelu(this->conv3_out, this->conv3_wb->pbias, this->prelu_gmma3->pdata);

    //flatten
    fullconnect(this->fc4_wb, this->conv3_out, this->fc4_out);
    prelu(this->fc4_out, this->fc4_wb->pbias, this->prelu_gmma4->pdata);

	//post conv process
    //conv51   score
    fullconnect(this->score_wb, this->fc4_out, this->score_);
    addbias(this->score_, this->score_wb->pbias);
    softmax(this->score_);
    //conv5_2   location
    fullconnect(this->location_wb, this->fc4_out, this->location_);
    addbias(this->location_, this->location_wb->pbias);
    // pBoxShow(location_);
}

Onet::Onet(){
    Othreshold = 0.8;
    this->rgb = new pBox;

	//conv1,prelu1
    this->conv1_matrix = new pBox;
    this->conv1_out = new pBox;
    this->pooling1_out = new pBox;

	//conv2,prelu2
    this->conv2_matrix = new pBox;
    this->conv2_out = new pBox;
    this->pooling2_out = new pBox;

	//conv3,prelu3
    this->conv3_matrix = new pBox;
    this->conv3_out = new pBox;
    this->pooling3_out = new pBox;

	//conv4,prelu4
    this->conv4_matrix = new pBox;
    this->conv4_out = new pBox;

    this->fc5_out = new pBox;

    this->score_ = new pBox;
    this->location_ = new pBox;
    this->keyPoint_ = new pBox;

	//weight
    this->conv1_wb = new Weight;
    this->prelu_gmma1 = new pRelu;//weight
    this->conv2_wb = new Weight;
    this->prelu_gmma2 = new pRelu;//weight
    this->conv3_wb = new Weight;
    this->prelu_gmma3 = new pRelu;//weight
    this->conv4_wb = new Weight;
    this->prelu_gmma4 = new pRelu;//weight
    this->fc5_wb = new Weight;
    this->prelu_gmma5 = new pRelu;//weight
    this->score_wb = new Weight;
    this->location_wb = new Weight;
    this->keyPoint_wb = new Weight;//weight

    // //                               w            sc  lc   ks s  lp  rp
    long conv1 = initConvAndFc(   this->conv1_wb,    32, 3,   3, 1, 0,  0);
    initpRelu(this->prelu_gmma1, 32);
    long conv2 = initConvAndFc(   this->conv2_wb,    64, 32,  3, 1, 0,  0);
    initpRelu(this->prelu_gmma2, 64);
    long conv3 = initConvAndFc(   this->conv3_wb,    64, 64,  3, 1, 0,  0);
    initpRelu(this->prelu_gmma3, 64);
    long conv4 = initConvAndFc(   this->conv4_wb,    128,64,  2, 1, 0,  0);
    initpRelu(this->prelu_gmma4, 128);
    long fc5 = initConvAndFc(     this->fc5_wb,      256,1152,1, 1, 0,  0);
    initpRelu(this->prelu_gmma5, 256);
    long score = initConvAndFc(   this->score_wb,    2,  256, 1, 1, 0,  0);
    long location = initConvAndFc(this->location_wb, 4,  256, 1, 1, 0,  0);
    long keyPoint = initConvAndFc(this->keyPoint_wb, 10, 256, 1, 1, 0,  0);
    long dataNumber[21] = {conv1,32,32, conv2,64,64, conv3,64,64, conv4,128,128, fc5,256,256, score,2, location,4, keyPoint,10};
    mydataFmt *pointTeam[21] = {this->conv1_wb->pdata, this->conv1_wb->pbias, this->prelu_gmma1->pdata, \
                                this->conv2_wb->pdata, this->conv2_wb->pbias, this->prelu_gmma2->pdata, \
                                this->conv3_wb->pdata, this->conv3_wb->pbias, this->prelu_gmma3->pdata, \
                                this->conv4_wb->pdata, this->conv4_wb->pbias, this->prelu_gmma4->pdata, \
                                this->fc5_wb->pdata, this->fc5_wb->pbias, this->prelu_gmma5->pdata, \
                                this->score_wb->pdata, this->score_wb->pbias, \
                                this->location_wb->pdata, this->location_wb->pbias, \
                                this->keyPoint_wb->pdata, this->keyPoint_wb->pbias \
                                };
    string filename = "Onet.bin";
	printf("Create Onet,Read Onet.bin ,21 pointers \n");
    readData(filename, dataNumber, pointTeam, 21);

    //Init the network
	printf("Onet buffer init\n");
	
	
	//change image to pBox format
    OnetImage2MatrixInit(rgb);

	//conv1,prelu1
    feature2MatrixInit(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolutionInit(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);
    //pool1
	maxPoolingInit(this->conv1_out, this->pooling1_out, 3, 2);
	//conv2,prelu2
    feature2MatrixInit(this->pooling1_out, this->conv2_matrix, this->conv2_wb);
    convolutionInit(this->conv2_wb, this->pooling1_out, this->conv2_out, this->conv2_matrix);
    //pool2
	maxPoolingInit(this->conv2_out, this->pooling2_out, 3, 2);
	//conv3,prelu3
    feature2MatrixInit(this->pooling2_out, this->conv3_matrix, this->conv3_wb);
    convolutionInit(this->conv3_wb, this->pooling2_out, this->conv3_out, this->conv3_matrix);
    //pool3
	maxPoolingInit(this->conv3_out, this->pooling3_out, 2, 2);
	//conv4,prelu4
    feature2MatrixInit(this->pooling3_out, this->conv4_matrix, this->conv4_wb);
    convolutionInit(this->conv4_wb, this->pooling3_out, this->conv4_out, this->conv4_matrix);
	//post conv precess
    fullconnectInit(this->fc5_wb, this->fc5_out);
    fullconnectInit(this->score_wb, this->score_);
    fullconnectInit(this->location_wb, this->location_);
    fullconnectInit(this->keyPoint_wb, this->keyPoint_);
}
Onet::~Onet(){
	printf("Free Onet\n");
    freepBox(this->rgb);
	//conv1,prelu1
    freepBox(this->conv1_matrix);
    freepBox(this->conv1_out);
	//pool1
    freepBox(this->pooling1_out);
	//conv2,prelu2
    freepBox(this->conv2_matrix);
    freepBox(this->conv2_out);
	//pool2
    freepBox(this->pooling2_out);
	//conv3,prelu3
    freepBox(this->conv3_matrix);
    freepBox(this->conv3_out);
	//pool3
    freepBox(this->pooling3_out);
	//conv4,prelu4
    freepBox(this->conv4_matrix);
    freepBox(this->conv4_out);
	//post conv process
    freepBox(this->fc5_out);
    freepBox(this->score_);
    freepBox(this->location_);
    freepBox(this->keyPoint_);

	//weight
    freeWeight(this->conv1_wb);
    freepRelu(this->prelu_gmma1);//weight
    freeWeight(this->conv2_wb);
    freepRelu(this->prelu_gmma2);//weight
    freeWeight(this->conv3_wb);
    freepRelu(this->prelu_gmma3);//weight
    freeWeight(this->conv4_wb);
    freepRelu(this->prelu_gmma4);//weight
    freeWeight(this->fc5_wb);
    freepRelu(this->prelu_gmma5);//weight
    freeWeight(this->score_wb);
    freeWeight(this->location_wb);
    freeWeight(this->keyPoint_wb);//weight
}
void Onet::OnetImage2MatrixInit(struct pBox *pbox){
    pbox->channel = 3;
    pbox->height = 48;
    pbox->width = 48;
    
    pbox->pdata = (mydataFmt *)malloc(pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
    if(pbox->pdata==NULL)cout<<"the image2MatrixInit is failed!!"<<endl;
    memset(pbox->pdata, 0, pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
}
void Onet::run(Mat &image){
	printf("Run Onet\n");
    image2Matrix(image, this->rgb);

	//conv1,prelu1
    feature2Matrix(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolution(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);
    prelu(this->conv1_out, this->conv1_wb->pbias, this->prelu_gmma1->pdata);

    //pool1
    maxPooling(this->conv1_out, this->pooling1_out, 3, 2);

	//conv1,prelu2
    feature2Matrix(this->pooling1_out, this->conv2_matrix, this->conv2_wb);
    convolution(this->conv2_wb, this->pooling1_out, this->conv2_out, this->conv2_matrix);
    prelu(this->conv2_out, this->conv2_wb->pbias, this->prelu_gmma2->pdata);
	
	//pool2
    maxPooling(this->conv2_out, this->pooling2_out, 3, 2);

    //conv3,prelu3
    feature2Matrix(this->pooling2_out, this->conv3_matrix, this->conv3_wb);
    convolution(this->conv3_wb, this->pooling2_out, this->conv3_out, this->conv3_matrix);
    prelu(this->conv3_out, this->conv3_wb->pbias, this->prelu_gmma3->pdata);
	
	//pool3
    maxPooling(this->conv3_out, this->pooling3_out, 2, 2);

    //conv4
    feature2Matrix(this->pooling3_out, this->conv4_matrix, this->conv4_wb);
    convolution(this->conv4_wb, this->pooling3_out, this->conv4_out, this->conv4_matrix);
    prelu(this->conv4_out, this->conv4_wb->pbias, this->prelu_gmma4->pdata);

	//post conv process
    fullconnect(this->fc5_wb, this->conv4_out, this->fc5_out);
    prelu(this->fc5_out, this->fc5_wb->pbias, this->prelu_gmma5->pdata);
    //conv6_1   score
    fullconnect(this->score_wb, this->fc5_out, this->score_);
    addbias(this->score_, this->score_wb->pbias);
    softmax(this->score_);
    // pBoxShow(this->score_);
    //conv6_2   location
    fullconnect(this->location_wb, this->fc5_out, this->location_);
    addbias(this->location_, this->location_wb->pbias);
    // pBoxShow(location_);
    //conv6_2   location
    fullconnect(this->keyPoint_wb, this->fc5_out, this->keyPoint_);
    addbias(this->keyPoint_, this->keyPoint_wb->pbias);
    // pBoxShow(keyPoint_);
}


mtcnn::mtcnn(int row, int col){
	printf("Create mtcnn,simple face\n");
    nms_threshold[0] = 0.7;
    nms_threshold[1] = 0.7;
    nms_threshold[2] = 0.7;

    float minl = row>col?row:col;
    int MIN_DET_SIZE = 12;
    int minsize = 60;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = 0.709;
    int factor_count = 0;

    while(minl>MIN_DET_SIZE){
        if(factor_count>0)m = m*factor;
        scales_.push_back(m);
        minl *= factor;
        factor_count++;
    }
    float minside = row<col ? row : col;
    int count = 0;
    for (vector<float>::iterator it = scales_.begin(); it != scales_.end(); it++){
        if (*it > 1){
            cout << "the minsize is too small" << endl;
            while (1);
        }
        if (*it < (MIN_DET_SIZE / minside)){
            scales_.resize(count);
            break;
        }
        count++;
    }
    simpleFace_ = new Pnet[scales_.size()];
}

mtcnn::~mtcnn(){
	printf("Free mtcnn,delete simpleFace\n");
    delete []simpleFace_;
}

void mtcnn::findFace(Mat &image){
	printf("Start find Face function\n");
    struct orderScore order;
    int count = 0;
    for (size_t i = 0; i < scales_.size(); i++) {
        int changedH = (int)ceil(image.rows*scales_.at(i));
        int changedW = (int)ceil(image.cols*scales_.at(i));
        resize(image, reImage, Size(changedW, changedH), 0, 0, cv::INTER_LINEAR);
        simpleFace_[i].run(reImage, scales_.at(i));
        nms(simpleFace_[i].boundingBox_, simpleFace_[i].bboxScore_, simpleFace_[i].nms_threshold);

        for(vector<struct Bbox>::iterator it=simpleFace_[i].boundingBox_.begin(); it!=simpleFace_[i].boundingBox_.end();it++){
            if((*it).exist){
                firstBbox_.push_back(*it);
                order.score = (*it).score;
                order.oriOrder = count;
                firstOrderScore_.push_back(order);
                count++;
            }
        }
        simpleFace_[i].bboxScore_.clear();
        simpleFace_[i].boundingBox_.clear();
    }
    //the first stage's nms
    if(count<1)return;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
    refineAndSquareBbox(firstBbox_, image.rows, image.cols);

    //second stage
    count = 0;
    for(vector<struct Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        if((*it).exist){
            Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            Mat secImage;
            resize(image(temp), secImage, Size(24, 24), 0, 0, cv::INTER_LINEAR);
            refineNet.run(secImage);
            if(*(refineNet.score_->pdata+1)>refineNet.Rthreshold){
                memcpy(it->regreCoord, refineNet.location_->pdata, 4*sizeof(mydataFmt));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(refineNet.score_->pdata+1);
                secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                secondBboxScore_.push_back(order);
            }
            else{
                (*it).exist=false;
            }
        }
    }
    if(count<1)return;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
    refineAndSquareBbox(secondBbox_, image.rows, image.cols);

    //third stage 
    count = 0;
    for(vector<struct Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            Mat thirdImage;
            resize(image(temp), thirdImage, Size(48, 48), 0, 0, cv::INTER_LINEAR);
            outNet.run(thirdImage);
            mydataFmt *pp=NULL;
            if(*(outNet.score_->pdata+1)>outNet.Othreshold){
                memcpy(it->regreCoord, outNet.location_->pdata, 4*sizeof(mydataFmt));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(outNet.score_->pdata+1);
                pp = outNet.keyPoint_->pdata;
                for(int num=0;num<5;num++){
                    (it->ppoint)[num] = it->y1 + (it->y2 - it->y1)*(*(pp+num));
                }
                for(int num=0;num<5;num++){
                    (it->ppoint)[num+5] = it->x1 + (it->x2 - it->x1)*(*(pp+num+5));
                }
                thirdBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                thirdBboxScore_.push_back(order);
            }
            else{
                it->exist=false;
            }
        }
    }

    if(count<1)return;
    refineAndSquareBbox(thirdBbox_, image.rows, image.cols);
    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
    for(vector<struct Bbox>::iterator it=thirdBbox_.begin(); it!=thirdBbox_.end();it++){
        if((*it).exist){
            rectangle(image, Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2), Scalar(0,0,255), 2,8,0);
            for(int num=0;num<5;num++)circle(image,Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,Scalar(0,255,255), -1);
        }
    }
    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();
	printf("Done find Face function\n");
}
