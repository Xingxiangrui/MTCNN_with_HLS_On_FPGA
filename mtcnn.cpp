#include "mtcnn.hpp"

Pnet::Pnet(){
    Pthreshold = 0.9;//0.6;
//    nms_threshold = 0.5;
    firstFlag = true;
	//in order of [weight] [input] [output]
    
//the RGB image in pBox format 
    this->rgb = new pBox;
//conv1,prelu1
    this->conv1_wb = new Weight;
    this->conv1_out = new pBox;//output
    this->conv1_out_pad = new pBox;//output padding
//pool1
    this->pool1_conv1_wb = new Weight;
    this->pool1_conv1_out = new pBox;//output
//conv2,relu2
    this->conv2_wb = new Weight;
    this->conv2_out = new pBox;//output
//conv3,relu3
    this->conv3_wb = new Weight;
    this->conv3_out = new pBox;//output
//post conv layers
    this->conv4c1_wb = new Weight;
    this->conv4c2_wb = new Weight;//weight
//    this->score_matrix = new pBox;
    this->score_ = new pBox;
//    this->location_matrix = new pBox;
    this->location_ = new pBox;    
    
    
    //weight pointer create,network initialize
    //                                 w           oc  ic  ks s lp  rp 
    long conv1 = initConvAndFc(   this->conv1_wb,  10, 3,  3, 1, 0, 0);
long pool1_conv1=initConvAndFc(this->pool1_conv1_wb,16,10, 3, 2, 0, 0);
    long conv2 = initConvAndFc(   this->conv2_wb,  32, 16, 3, 1, 0, 0);
    long conv3 = initConvAndFc(   this->conv3_wb,  32, 32, 3, 1, 0, 0);
    long conv4c1 = initConvAndFc(this->conv4c1_wb, 2,  32, 1, 1, 0, 0);
    long conv4c2 = initConvAndFc(this->conv4c2_wb, 4,  32, 1, 1, 0, 0);
    long dataNumber[12] = {conv1,10,pool1_conv1,16,conv2,32,conv3,32,conv4c1,2,conv4c2,4};
    mydataFmt *pointTeam[12] = {this->conv1_wb->pdata, this->conv1_wb->pbias, \
                            this->pool1_conv1_wb->pdata, this->pool1_conv1_wb->pbias, \
                            this->conv2_wb->pdata, this->conv2_wb->pbias, \
                            this->conv3_wb->pdata, this->conv3_wb->pbias, \
                            this->conv4c1_wb->pdata, this->conv4c1_wb->pbias, \
                            this->conv4c2_wb->pdata, this->conv4c2_wb->pbias \
                            };
    string filename = "Pnet.bin";
	printf("Create Pnet,Read Pnet.bin ,12 pointers \n");
    readData(filename, dataNumber, pointTeam, 12);
}
Pnet::~Pnet(){
	//in order of [weight] [input] [output]
//input RGB image
    freepBox(this->rgb);
//conv1
    freepBox(this->conv1_out);//output
    freepBox(this->conv1_out_pad);
//pool1
    freepBox(this->pool1_conv1_out);
//conv2
    freepBox(this->conv2_out);//output
//conv3
    freepBox(this->conv3_out);//output
//fc
    freepBox(this->score_);
    freepBox(this->location_);
//    freepBox(this->score_matrix);
//    freepBox(this->location_matrix);    
//post conv layers
    freeWeight(this->conv1_wb);
    freeWeight(this->pool1_conv1_wb);
    freeWeight(this->conv2_wb);
    freeWeight(this->conv3_wb);
    freeWeight(this->conv4c1_wb);
    freeWeight(this->conv4c2_wb);//weight

    
	printf("Free Pnet\n");
}

void Pnet::run(Mat &image, float scale){
	printf("Start run Pnet\n");
    if(firstFlag){
        printf("Pnet buffer init\n");
    //change Mat image to pBox format
        image2MatrixInit(image, this->rgb);
		//conv1,prelu1
        convolutionInit(this->conv1_wb, this->rgb, this->conv1_out);
        featurePadInit(this->conv1_out, this->conv1_out_pad, 0,1);
		//pool1
        convolutionInit(this->pool1_conv1_wb, this->conv1_out_pad, this->pool1_conv1_out);
    //conv2,prelu2
        convolutionInit(this->conv2_wb, this->pool1_conv1_out, this->conv2_out);  
    //conv3,prelu3
        convolutionInit(this->conv3_wb, this->conv2_out, this->conv3_out);
    //post conv layers
        convolutionInit(this->conv4c1_wb, this->conv3_out, this->score_);
        convolutionInit(this->conv4c2_wb, this->conv3_out, this->location_);
        firstFlag = false;
    }

	 //change Mat image to pBox format
    image2Matrix(image, this->rgb);
//    cout << "############   this->rgb   #############" << endl;
//    pBoxShow(this->rgb);

	 //conv1,relu1
    convolution_3x3(this->rgb->height, this->rgb->width, this->conv1_wb->in_ChannelNum,
     this->conv1_out->height, this->conv1_out->width, this->conv1_wb->out_ChannelNum,
     this->conv1_wb->stride, this->conv1_wb->pdata, this->rgb->pdata, this->conv1_out->pdata,
     this->conv1_wb->pbias);
//    convolution(this->conv1_wb, this->rgb, this->conv1_out);
//    relu(this->conv1_out, this->conv1_wb->pbias);
//    pBoxShow(this->conv1_out);
    //add padding
//    cout << "##################pad###############" << endl;
    featurePad(this->conv1_out, this->conv1_out_pad, 0,1);
//    pBoxShow(this->conv1_out_pad);
	 //pool1_conv1,pool1_relu
    convolution_3x3(this->conv1_out_pad->height, this->conv1_out_pad->width, 
        this->pool1_conv1_wb->in_ChannelNum, this->pool1_conv1_out->height, this->pool1_conv1_out->width,
        this->pool1_conv1_wb->out_ChannelNum, this->pool1_conv1_wb->stride, this->pool1_conv1_wb->pdata,
        this->conv1_out_pad->pdata, this->pool1_conv1_out->pdata, this->pool1_conv1_wb->pbias);
    // convolution(this->pool1_conv1_wb, this->conv1_out_pad, this->pool1_conv1_out);
 //   relu(this->pool1_conv1_out, this->pool1_conv1_wb->pbias);  
//    pBoxShow(this->pool1_conv1_out);
    
	 //conv2,relu2
    convolution_3x3(this->pool1_conv1_out->height, this->pool1_conv1_out->width,
        this->conv2_wb->in_ChannelNum, this->conv2_out->height, this->conv2_out->width,
        this->conv2_wb->out_ChannelNum, this->conv2_wb->stride, this->conv2_wb->pdata,
        this->pool1_conv1_out->pdata, this->conv2_out->pdata, this->conv2_wb->pbias);
    // convolution(this->conv2_wb, this->pool1_conv1_out, this->conv2_out);
    //relu(this->conv2_out, this->conv2_wb->pbias);
    
    //conv3,relu3
    convolution_3x3(this->conv2_out->height, this->conv2_out->width, this->conv3_wb->in_ChannelNum,
        this->conv3_out->height, this->conv3_out->width, this->conv3_wb->out_ChannelNum,
        this->conv3_wb->stride, this->conv3_wb->pdata, this->conv2_out->pdata, this->conv3_out->pdata,
        this->conv3_wb->pbias);
    // convolution(this->conv3_wb, this->conv2_out, this->conv3_out);
    //relu(this->conv3_out, this->conv3_wb->pbias);
    
//    pBoxShow(this->conv3_out);
    
    //conv4-1   score
    convolution(this->conv4c1_wb, this->conv3_out, this->score_);
    addbias(this->score_, this->conv4c1_wb->pbias);
    softmax(this->score_);
    
    
//    cout << "score_" << this->score_->pdata[0] <<" "<< this->score_->pdata[1] << endl;
//    pBoxShow(this->score_);
    //conv4-2   location
    convolution(this->conv4c2_wb, this->conv3_out, this->location_);
    addbias(this->location_, this->conv4c2_wb->pbias);
//    cout << "localtion " << endl;
//    cout << this->location_->pdata[0] <<" "<< this->location_->pdata[1] << endl;
//    cout << this->location_->pdata[2] <<" "<< this->location_->pdata[3] << endl;
//    pBoxShow(this->location_);
    
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
    Rthreshold = 0.8;//0.7;
    this->rgb = new pBox;
    this->rgb_pad = new pBox;
	//conv1
    this->conv1_out = new pBox;
    this->conv1_out_pad = new pBox;
	//pool1
    this->pool_conv1_out = new pBox;
    this->pool_conv1_out_pad = new pBox;
	//conv2
    this->conv2_out = new pBox;
    this->conv2_out_pad = new pBox;
	//pool2
    this->pool2_conv3_out = new pBox;
    this->pool2_conv3_out_pad = new pBox;
	//conv3
    this->conv3_out = new pBox;
    this->conv3_out_pad = new pBox;
	//post conv process
    this->fc4_out = new pBox;
    this->score_ = new pBox;
    this->location_ = new pBox;

	//weight
    this->conv1_wb = new Weight;
    this->pool_conv1_wb = new Weight;
    this->conv2_wb = new Weight;
    this->pool2_conv3_wb = new Weight;
    this->conv3_wb = new Weight;
    this->fc4_wb = new Weight;
    this->score_wb = new Weight;
    this->location_wb = new Weight;//weight
    //                                    w         sc  lc   ks s  lp  rp
    long conv1 = initConvAndFc(   this->conv1_wb,   28, 3,   3, 1, 0,  0);
long pool_conv1=initConvAndFc(this->pool_conv1_wb,  28, 28,  3, 2, 0,  0);
    long conv2 = initConvAndFc(   this->conv2_wb,   48, 28,  3, 1, 0,  0);
long pool2_conv3=initConvAndFc(this->pool2_conv3_wb,48, 48,  3, 2, 0,  0);
    long conv3 = initConvAndFc(   this->conv3_wb,   64, 48,  3, 2, 0,  0);
    long fc4 = initConvAndFc(     this->fc4_wb,     128,576, 1, 1, 0,  0);
    long score = initConvAndFc(   this->score_wb,   2,  128, 1, 1, 0,  0);
    long location = initConvAndFc(this->location_wb,4,  128, 1, 1, 0,  0);
    long dataNumber[16] = {conv1,28,pool_conv1,28,conv2,48,pool2_conv3,48,conv3,64,fc4,128,score,2,location,4};
    mydataFmt *pointTeam[16] = {this->conv1_wb->pdata, this->conv1_wb->pbias, \
                                this->pool_conv1_wb->pdata, this->pool_conv1_wb->pbias, \
                                this->conv2_wb->pdata, this->conv2_wb->pbias, \
                                this->pool2_conv3_wb->pdata, this->pool2_conv3_wb->pbias, \
                                this->conv3_wb->pdata, this->conv3_wb->pbias, \
                                this->fc4_wb->pdata, this->fc4_wb->pbias, \
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
    featurePadInit(this->rgb, this->rgb_pad, 1, 1);
	  //conv1
    convolutionInit(this->conv1_wb, this->rgb_pad, this->conv1_out);
    featurePadInit(this->conv1_out, this->conv1_out_pad, 0, 1);
    //pool1
    convolutionInit(this->pool_conv1_wb, this->conv1_out_pad, this->pool_conv1_out);
    featurePadInit(this->pool_conv1_out, this->pool_conv1_out_pad, 1, 1);
	  //conv2
    convolutionInit(this->conv2_wb, this->pool_conv1_out_pad, this->conv2_out);
    featurePadInit(this->conv2_out, this->conv2_out_pad, 0, 1);
    //pool2
    convolutionInit(this->pool2_conv3_wb, this->conv2_out_pad, this->pool2_conv3_out);
    featurePadInit(this->pool2_conv3_out, this->pool2_conv3_out_pad, 0, 1);
    //conv3
    convolutionInit(this->conv3_wb, this->pool2_conv3_out_pad, this->conv3_out);
//    featurePadInit(this->conv3_out, this->conv3_out_pad, 0, 1);
    //post conv precess
	  fullconnectInit(this->fc4_wb, this->fc4_out);
    fullconnectInit(this->score_wb, this->score_);
    fullconnectInit(this->location_wb, this->location_);
}
Rnet::~Rnet(){
	printf("Free Rnet\n");
    freepBox(this->rgb);
    freepBox(this->rgb_pad);;
	//conv1
    freepBox(this->conv1_out);
    freepBox(this->conv1_out_pad);
	//pool1
    freepBox(this->pool_conv1_out);
    freepBox(this->pool_conv1_out_pad);
	//conv2
    freepBox(this->conv2_out);
    freepBox(this->conv2_out_pad);
	//pool2
    freepBox(this->pool2_conv3_out);
    freepBox(this->pool2_conv3_out_pad);
	//conv3
    freepBox(this->conv3_out);
//    freepBox(this->conv3_out_pad);
	
	//post conv process
    freepBox(this->fc4_out);
    freepBox(this->score_);
    freepBox(this->location_);

	//weight
    freeWeight(this->conv1_wb);
    freeWeight(this->pool_conv1_wb);
    freeWeight(this->conv2_wb);
    freeWeight(this->pool2_conv3_wb);
    freeWeight(this->conv3_wb);
    freeWeight(this->fc4_wb);
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
//    cout << "########### rgb ##########" << endl;
//    pBoxShow(this->rgb);
    //conv1
    featurePad(this->rgb, this->rgb_pad, 1, 1);
//    pBoxShow(this->rgb_pad);
    convolution_3x3(this->rgb_pad->height, this->rgb_pad->width, this->conv1_wb->in_ChannelNum,
        this->conv1_out->height, this->conv1_out->width, this->conv1_wb->out_ChannelNum,
        this->conv1_wb->stride, this->conv1_wb->pdata, this->rgb_pad->pdata, this->conv1_out->pdata,
        this->conv1_wb->pbias);
//  convolution(this->conv1_wb, this->rgb_pad, this->conv1_out);
 //   relu(this->conv1_out, this->conv1_wb->pbias);
//    pBoxShow(this->conv1_out);
    featurePad(this->conv1_out, this->conv1_out_pad, 0, 1);//24+2,24+2,48
    
    //pool1
    convolution_3x3(this->conv1_out_pad->height, this->conv1_out_pad->width, this->pool_conv1_wb->in_ChannelNum,
        this->pool_conv1_out->height, this->pool_conv1_out->width, this->pool_conv1_wb->out_ChannelNum,
        this->pool_conv1_wb->stride, this->pool_conv1_wb->pdata, this->conv1_out_pad->pdata,
        this->pool_conv1_out->pdata, this->pool_conv1_wb->pbias);
//  convolution(this->pool_conv1_wb, this->conv1_out_pad, this->pool_conv1_out);
  //  relu(this->pool_conv1_out, this->pool_conv1_wb->pbias);
    featurePad(this->pool_conv1_out, this->pool_conv1_out_pad, 1, 1);
    
    //conv2,relu2
    convolution_3x3(this->pool_conv1_out_pad->height, this->pool_conv1_out_pad->width, this->conv2_wb->in_ChannelNum,
        this->conv2_out->height, this->conv2_out->width, this->conv2_wb->out_ChannelNum, this->conv2_wb->stride,
        this->conv2_wb->pdata, this->pool_conv1_out_pad->pdata, this->conv2_out->pdata,
        this->conv2_wb->pbias);
//  convolution(this->conv2_wb, this->pool_conv1_out_pad, this->conv2_out);
//   relu(this->conv2_out, this->conv2_wb->pbias);
    featurePad(this->conv2_out, this->conv2_out_pad, 0, 1);
    
    //pool2
    convolution_3x3(this->conv2_out_pad->height, this->conv2_out_pad->width, this->pool2_conv3_wb->in_ChannelNum,
        this->pool2_conv3_out->height, this->pool2_conv3_out->width, this->pool2_conv3_wb->out_ChannelNum,
        this->pool2_conv3_wb->stride, this->pool2_conv3_wb->pdata, this->conv2_out_pad->pdata,
        this->pool2_conv3_out->pdata, this->pool2_conv3_wb->pbias);
// convolution(this->pool2_conv3_wb, this->conv2_out_pad, this->pool2_conv3_out);
//    relu(this->pool2_conv3_out, this->pool2_conv3_wb->pbias);
    featurePad(this->pool2_conv3_out, this->pool2_conv3_out_pad, 0, 1);
    
    //conv3
    convolution_3x3(this->pool2_conv3_out_pad->height, this->pool2_conv3_out_pad->width, this->conv3_wb->in_ChannelNum,
        this->conv3_out->height, this->conv3_out->width, this->conv3_wb->out_ChannelNum, this->conv3_wb->stride,
        this->conv3_wb->pdata, this->pool2_conv3_out_pad->pdata, this->conv3_out->pdata,
        this->conv3_wb->pbias);
//   convolution(this->conv3_wb, this->pool2_conv3_out_pad, this->conv3_out);
//    relu(this->conv3_out, this->conv3_wb->pbias);
//    pBoxShow(this->conv3_out);
//    featurePad(this->conv3_out, this->conv3_out_pad, 1, 1);
//    for(int n=0; n<576; n++){
//        cout << this->conv3_out->pdata[n] << endl;
//    }    

    //flatten conv4
    fullconnect(this->fc4_wb, this->conv3_out, this->fc4_out);
    relu(this->fc4_out, this->fc4_wb->pbias);
//    for(int n=0; n<576; n++){
//        cout << this->fc4_out->pdata[n] << endl;
////        cout << this->fc4_wb->pdata[n] << endl;
//    }
//    pBoxShow(this->fc4_out);
	//post conv process
    //conv51   score
    fullconnect(this->score_wb, this->fc4_out, this->score_);
    addbias(this->score_, this->score_wb->pbias);
    softmax(this->score_);
//    cout << "score " << this->score_->pdata[0] << " " << this->score_->pdata[1] << endl;
    
    //conv52   location
    fullconnect(this->location_wb, this->fc4_out, this->location_);
    addbias(this->location_, this->location_wb->pbias);
//    cout << "location " << this->location_->pdata[0] << " " << this->location_->pdata[1] << endl;
//    cout << "location " << this->location_->pdata[2] << " " << this->location_->pdata[3] << endl;
}

Onet::Onet(){
    Othreshold = 0.8;
    this->rgb = new pBox;
    this->rgb_pad = new pBox;
    //conv1,relu1
    this->conv1_out = new pBox;
    this->conv1_out_pad = new pBox;
    //conv2,relu2
    this->conv2_out = new pBox;
    this->conv2_out_pad = new pBox;
    //conv3,relu3
    this->conv3_out = new pBox;
    this->conv3_out_pad = new pBox;
    //conv4_,relu4_
    this->conv4_out = new pBox;
    this->conv4_out_pad = new pBox;
    //conv5_,relu5_
    this->conv5_out = new pBox;
    this->conv5_out_pad = new pBox;
    //conv6_,relu6_
    this->conv6_out = new pBox;
    
    this->fc5_out = new pBox;

    this->score_ = new pBox;
    this->location_ = new pBox;
//    this->keyPoint_ = new pBox;

	//weight
    this->conv1_wb = new Weight;
    this->conv2_wb = new Weight;
    this->conv3_wb = new Weight;
    this->conv4_wb = new Weight;
    this->conv5_wb = new Weight;
    this->conv6_wb = new Weight;
    this->fc5_wb = new Weight;//tensorflow name:conv5
    this->score_wb = new Weight;
    this->location_wb = new Weight;
//    this->keyPoint_wb = new Weight;//weight
//                                         w         sc  lc   ks s  lp  rp
    long conv1 = initConvAndFc(   this->conv1_wb,    32, 3,   3, 1, 0,  0);
    long conv2 = initConvAndFc(   this->conv2_wb,    32, 32,  3, 2, 0,  0);
    long conv3 = initConvAndFc(   this->conv3_wb,    64, 32,  3, 1, 0,  0);
    long conv4_ = initConvAndFc(  this->conv4_wb,    64, 64,  3, 2, 0,  0);
    long conv5_ = initConvAndFc(  this->conv5_wb,    128,64,  3, 2, 0,  0);
    long conv6_ = initConvAndFc(  this->conv6_wb,    128,128, 3, 2, 0,  0);
    long fc5 = initConvAndFc(     this->fc5_wb,      256,1152,1, 1, 0,  0);
    long score = initConvAndFc(   this->score_wb,    2,  256, 1, 1, 0,  0);
    long location = initConvAndFc(this->location_wb, 4,  256, 1, 1, 0,  0);
//    long keyPoint = initConvAndFc(this->keyPoint_wb, 10, 256, 1, 1, 0,  0);
    long dataNumber[18] = {conv1,32,conv2,32,conv3,64,conv4_,64,conv5_,128,conv6_,128,fc5,256,score,2,location,4};
    mydataFmt *pointTeam[18] = {this->conv1_wb->pdata, this->conv1_wb->pbias, \
                                this->conv2_wb->pdata, this->conv2_wb->pbias, \
                                this->conv3_wb->pdata, this->conv3_wb->pbias, \
                                this->conv4_wb->pdata, this->conv4_wb->pbias,\
                                this->conv5_wb->pdata, this->conv5_wb->pbias,\
                                this->conv6_wb->pdata, this->conv6_wb->pbias,\
                                this->fc5_wb->pdata,   this->fc5_wb->pbias, \
                                this->score_wb->pdata, this->score_wb->pbias, \
                                this->location_wb->pdata, this->location_wb->pbias};
    string filename = "Onet.bin";
	  printf("Create Onet,Read Onet.bin ,18 pointers \n");
    readData(filename, dataNumber, pointTeam, 18);

    //Init the network
	printf("Onet buffer init\n");
	
	  //change image to pBox format
    OnetImage2MatrixInit(rgb);
    featurePadInit(this->rgb, this->rgb_pad, 1, 1);//s=1
	  //conv1
    convolutionInit(this->conv1_wb, this->rgb_pad, this->conv1_out);
    featurePadInit(this->conv1_out, this->conv1_out_pad, 0, 1);//s=2
	  //conv2
    convolutionInit(this->conv2_wb, this->conv1_out_pad, this->conv2_out);
    featurePadInit(this->conv2_out, this->conv2_out_pad, 1, 1);//s=1
    //conv3
    convolutionInit(this->conv3_wb, this->conv2_out_pad, this->conv3_out);
    featurePadInit(this->conv3_out, this->conv3_out_pad, 0, 1);//s=2
    //conv4_
    convolutionInit(this->conv4_wb, this->conv3_out_pad, this->conv4_out);
    featurePadInit(this->conv4_out, this->conv4_out_pad, 0, 1);//s=2
    //conv5_
    convolutionInit(this->conv5_wb, this->conv4_out_pad, this->conv5_out);
    featurePadInit(this->conv5_out, this->conv5_out_pad, 0, 1);//s=2
    //conv6_
    convolutionInit(this->conv6_wb, this->conv5_out_pad, this->conv6_out);
    //post conv precess
	  fullconnectInit(this->fc5_wb, this->fc5_out);
    fullconnectInit(this->score_wb, this->score_);
    fullconnectInit(this->location_wb, this->location_);
}
Onet::~Onet(){
    printf("Free Onet\n");
    freepBox(this->rgb);
    freepBox(this->rgb_pad);
    //conv1
    freepBox(this->conv1_out);
    freepBox(this->conv1_out_pad);
    //conv2
    freepBox(this->conv2_out);
    freepBox(this->conv2_out_pad);
    //conv3
    freepBox(this->conv3_out);
    freepBox(this->conv3_out_pad);
    //conv4_
    freepBox(this->conv4_out);
    freepBox(this->conv4_out_pad);
    //conv5_
    freepBox(this->conv5_out);
    freepBox(this->conv5_out_pad);
    //conv6_
    freepBox(this->conv6_out);
    //post conv process
    freepBox(this->fc5_out);
    freepBox(this->score_);
    freepBox(this->location_);
//    freepBox(this->keyPoint_);

    //weight
    freeWeight(this->conv1_wb);
    freeWeight(this->conv2_wb);
    freeWeight(this->conv3_wb);
    freeWeight(this->conv4_wb);
    freeWeight(this->conv5_wb);
    freeWeight(this->conv6_wb);
    freeWeight(this->fc5_wb);
    freeWeight(this->score_wb);
    freeWeight(this->location_wb);
//    freeWeight(this->keyPoint_wb);//weight
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
//    pBoxShow(this->rgb);
    featurePad(this->rgb, this->rgb_pad, 1, 1);//s=1

	  //conv1,relu1
    convolution_3x3(this->rgb_pad->height, this->rgb_pad->width, this->conv1_wb->in_ChannelNum,
        this->conv1_out->height, this->conv1_out->width, this->conv1_wb->out_ChannelNum, this->conv1_wb->stride,
        this->conv1_wb->pdata, this->rgb_pad->pdata, this->conv1_out->pdata,
        this->conv1_wb->pbias);
//  convolution(this->conv1_wb, this->rgb_pad, this->conv1_out);
//    relu(this->conv1_out, this->conv1_wb->pbias);
//    pBoxShow(this->conv1_out);
    featurePad(this->conv1_out, this->conv1_out_pad, 0, 1);//s=2
    
	  //conv2,relu2
    convolution_3x3(this->conv1_out_pad->height, this->conv1_out_pad->width, this->conv2_wb->in_ChannelNum,
        this->conv2_out->height, this->conv2_out->width, this->conv2_wb->out_ChannelNum,this->conv2_wb->stride,
        this->conv2_wb->pdata, this->conv1_out_pad->pdata, this->conv2_out->pdata,
        this->conv2_wb->pbias);
//  convolution(this->conv2_wb, this->conv1_out_pad, this->conv2_out);
//    relu(this->conv2_out, this->conv2_wb->pbias);
//    pBoxShow(this->conv2_out);
    featurePad(this->conv2_out, this->conv2_out_pad, 1, 1);//s=1

    //conv3,relu3
    convolution_3x3(this->conv2_out_pad->height, this->conv2_out_pad->width, this->conv3_wb->in_ChannelNum,
        this->conv3_out->height, this->conv3_out->width, this->conv3_wb->out_ChannelNum, this->conv3_wb->stride,
        this->conv3_wb->pdata, this->conv2_out_pad->pdata, this->conv3_out->pdata,
        this->conv3_wb->pbias);
//  convolution(this->conv3_wb, this->conv2_out_pad, this->conv3_out);
//    relu(this->conv3_out, this->conv3_wb->pbias);
//    pBoxShow(this->conv3_out);
    featurePad(this->conv3_out, this->conv3_out_pad, 0, 1);//s=2

    //conv4_,relu4
    convolution_3x3(this->conv3_out_pad->height, this->conv3_out_pad->width, this->conv4_wb->in_ChannelNum,
        this->conv4_out->height, this->conv4_out->width, this->conv4_wb->out_ChannelNum, this->conv4_wb->stride,
        this->conv4_wb->pdata, this->conv3_out_pad->pdata, this->conv4_out->pdata,
        this->conv4_wb->pbias);
//  convolution(this->conv4_wb, this->conv3_out_pad, this->conv4_out);
//    relu(this->conv4_out, this->conv4_wb->pbias);
//    pBoxShow(this->conv4_out);
    featurePad(this->conv4_out, this->conv4_out_pad, 0, 1);//s=2
    
    //conv5_,relu5
    convolution_3x3(this->conv4_out_pad->height, this->conv4_out_pad->width, this->conv5_wb->in_ChannelNum,
        this->conv5_out->height, this->conv5_out->width, this->conv5_wb->out_ChannelNum, this->conv5_wb->stride,
        this->conv5_wb->pdata, this->conv4_out_pad->pdata, this->conv5_out->pdata,
        this->conv5_wb->pbias);
//  convolution(this->conv5_wb, this->conv4_out_pad, this->conv5_out);
//    relu(this->conv5_out, this->conv5_wb->pbias);
//    pBoxShow(this->conv5_out);
    featurePad(this->conv5_out, this->conv5_out_pad, 0, 1);//s=2
//    pBoxShow(this->conv5_out_pad);

    //conv6_,relu6
    convolution_3x3(this->conv5_out->height, this->conv5_out_pad->width, this->conv6_wb->in_ChannelNum,
        this->conv6_out->height, this->conv6_out->width, this->conv6_wb->out_ChannelNum, this->conv6_wb->stride,
        this->conv6_wb->pdata, this->conv5_out_pad->pdata, this->conv6_out->pdata,
        this->conv6_wb->pbias);
//    convolution(this->conv6_wb, this->conv5_out_pad, this->conv6_out);
//    for(int n = 0; n < 100; n++){
//        cout <<  this->conv6_wb->pdata[n] << endl;     
//    }
//    relu(this->conv6_out, this->conv6_wb->pbias);
//    pBoxShow(this->conv6_out);
    
	//post conv process
    fullconnect(this->fc5_wb, this->conv6_out, this->fc5_out);
    relu(this->fc5_out, this->fc5_wb->pbias);
//    pBoxShow(this->fc5_out);
    
    //conv6_1   score
    fullconnect(this->score_wb, this->fc5_out, this->score_);
    addbias(this->score_, this->score_wb->pbias);
    softmax(this->score_);
    cout << "score " << this->score_->pdata[0] << " " << this->score_->pdata[1] << endl;
    
    //pBoxShow(this->score_);
    //conv6_2   location
    fullconnect(this->location_wb, this->fc5_out, this->location_);
    addbias(this->location_, this->location_wb->pbias);
//    float sum = 0.0000;
//    for(int n = 0; n < 256; n++){
//        sum += this->location_wb->pdata[n] * this->fc5_out->pdata[n];
////        cout <<  this->location_wb->pdata[n] << endl;
//    }
//    sum += this->location_wb->pbias[0];
//    cout << "SUM " << sum << endl;
    
//    cout << this->location_wb->pbias[0] << this->location_wb->pbias[1] << endl;
//    cout << this->location_wb->pbias[2] << this->location_wb->pbias[3] << endl;
    cout << "location " << this->location_->pdata[0] << " " << this->location_->pdata[1] << endl;
    cout << "location " << this->location_->pdata[2] << " " << this->location_->pdata[3] << endl;

    //conv6_2   location
//    fullconnect(this->keyPoint_wb, this->fc5_out, this->keyPoint_);
//    addbias(this->keyPoint_, this->keyPoint_wb->pbias);
    // pBoxShow(keyPoint_);
}


mtcnn::mtcnn(int row, int col){
	printf("Create mtcnn,simple face\n");
    nms_threshold[0] = 0.7;
    nms_threshold[1] = 0.7;
    nms_threshold[2] = 0.7;

    float minl = row>col?row:col;
    int MIN_DET_SIZE = 12;
    int minsize = 30;//60;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = 0.85;//0.709;
    int factor_count = 0;

    while(minl>(MIN_DET_SIZE)){
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

int mtcnn::findFace(Mat &image){
    printf("------Start find Face function\n");
    struct orderScore order;
    int count = 0;
    
 
//scales 0.400000  h:116  w:142
//scales 0.340000  h:98  w:120
//scales 0.289000  h:84  w:102
//scales 0.245650  h:72  w:88
//scales 0.208803  h:62  w:74
//scales 0.177482  h:52  w:64
//scales 0.150860  h:44  w:54
//scales 0.128231  h:38  w:46
//scales 0.108996  h:32  w:40
//scales 0.092647  h:28  w:34
//scales 0.078750  h:24  w:28
//scales 0.066937  h:20  w:24
//scales 0.056897  h:18  w:22
//scales 0.048362  h:14  w:18
    
    
    
//PNet debug
//    Mat image_debug = imread("/1t_second/myzhuang2/MTCNN/FPGA-mtcnn/00.jpg");
////    cout << "########## input image 00.jpg  ##########" << endl;
//    int changedH = (int)ceil(image_debug.rows*scales_.at(1)/2);
//    int changedW = (int)ceil(image_debug.cols*scales_.at(1)/2);
//    printf("scales %f  h:%d  w:%d\n", scales_.at(1), changedH*2, changedW*2);
////    12: scales 0.056897  h:18  w:22
//    resize(image_debug, reImage, Size(changedW*2, changedH*2), 0, 0, cv::INTER_AREA);
////    imshow("image_debug", image_debug);
////    imwrite("image_debug98x120.jpg",reImage);
////	 image.release();
////     	waitKey(0);
//
//    cout << reImage.rows << " " << reImage.cols << endl;
////    int aaa = 0;
////    for(int channel=0; channel<3; channel++){
////        for(int y=0; y<24; y++){
////            for(int x=0; x<24; x++){
////                aaa = reImage.at<Vec3b>(y,x)[channel];
////                cout << aaa << " ";
////            }
////            cout << endl;
////        }
////        cout << endl;
////    }
//            
//    simpleFace_[1].run(reImage, scales_.at(1));
//    nms(simpleFace_[1].boundingBox_, simpleFace_[1].bboxScore_, simpleFace_[1].nms_threshold);
//    
//        for(vector<struct Bbox>::iterator it=simpleFace_[1].boundingBox_.begin(); it!=simpleFace_[1].boundingBox_.end();it++){
//            if((*it).exist){
//                firstBbox_.push_back(*it);
//                order.score = (*it).score;
//                order.oriOrder = count;
//                firstOrderScore_.push_back(order);
//                count++;
//            }
//        }
//        simpleFace_[1].bboxScore_.clear();
//        simpleFace_[1].boundingBox_.clear();
//    if(count<1)return 2;
//    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
//    refineAndSquareBbox(firstBbox_, image_debug.rows, image_debug.cols);
////    draw first stage            
//    for(vector<struct Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
//        if((*it).exist){
//            cout << "y1 " << (*it).y1 << " x1 " << (*it).x1 << " y2 " << (*it).y2 << " x2 " << (*it).x2 << endl;
////            rectangle(image, Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2), Scalar(255,0,0), 1,8,0);
//        }
//    }
//
//    cout <<"system pause" << endl; while(1);



       
//******************************* first stage *******************************
    for (size_t i = 0; i < scales_.size(); i++) {
        int changedH = (int)ceil(image.rows*scales_.at(i)/2);
        int changedW = (int)ceil(image.cols*scales_.at(i)/2);
        printf("scales %f  h:%d  w:%d\n", scales_.at(i), changedH*2, changedW*2);
        resize(image, reImage, Size(changedW*2, changedH*2), 0, 0, cv::INTER_AREA);//INTER_LINEAR
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
    if(count<1)return 2;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
    refineAndSquareBbox(firstBbox_, image.rows, image.cols);
//******************************* first stage *******************************



    
    
//    draw first stage            
//    for(vector<struct Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
//        if((*it).exist){
//            cout << "y1 " << (*it).y1 << " x1 " << (*it).x1 << " y2 " << (*it).y2 << " x2 " << (*it).x2 << endl;
//            rectangle(image, Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2), Scalar(255,0,0), 1,8,0);
//        }
//    }
//    rectangle(image, Point(199, 151), Point(264, 215), Scalar(255,0,0), 1,8,0);   
//    printf("PNET Bbox count %d\n", count);
//    imshow("result", image);
//    waitKey(0);
//    cout <<"system pause" << endl; while(1);
    
    
    
//RNet debug
//    Mat image_debug = imread("/1t_second/myzhuang2/MTCNN/FPGA-mtcnn/img24x24.jpg");
//    cout << "########## input image 00.jpg  ##########" << endl;    
//    for(vector<struct Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
//        if((*it).exist){
//            cout <<"y1 "<<(*it).y1<<" x1 "<<(*it).x1<<" y2 "<<(*it).y2<<" x2 "<<(*it).x2<<endl;
//        }
//    }
//    cout <<"system pause" << endl; while(1);  
//    vector<struct Bbox>::iterator it = firstBbox_.begin();
//    Rect temp(199, 151, (264-199), (215-151));
//    Mat secImage;
//    resize(image_debug(temp), secImage, Size(24, 24), 0, 0, cv::INTER_LINEAR); 
//    cout <<"y1 "<<(*it).y1<<" x1 "<<(*it).x1<<" y2 "<<(*it).y2<<" x2 "<<(*it).x2<<endl;
//    Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
//    Mat img24 = image_debug(temp);
//    imwrite("img24x24.jpg",secImage);
//    cout << img24.rows << " " << img24.cols << endl;

//    int aaa = 0;
//    for(int channel=0; channel<3; channel++){
//        for(int y=0; y<24; y++){
//            for(int x=0; x<24; x++){
//                aaa = image_debug.at<Vec3b>(y,x)[channel];
//                cout << aaa << "\t";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
//    refineNet.run(image_debug);
//    cout <<"system pause" << endl; while(1);
    
    

//****************************** second stage ******************************
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
    if(count<1)return 2;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
    refineAndSquareBbox(secondBbox_, image.rows, image.cols);
//****************************** second stage ******************************




//
////Onet debug
//    printf("############### Onet debug ###############\n\n");
//    Mat image_debug = imread("/1t_second/myzhuang2/MTCNN/FPGA-mtcnn/img48x48.jpg");
//    cout << "########## input image img48x48.jpg  ##########" << endl;
////    cout << image_debug.rows << " " << image_debug.cols << endl;
////    int aaa = 0;
////    for(int channel=0; channel<3; channel++){
////        for(int y=0; y<18; y++){
////            for(int x=0; x<22; x++){
////                aaa = image_debug.at<Vec3b>(y,x)[channel];
////                cout << aaa << " ";
////            }
////            cout << endl;
////        }
////        cout << endl;
////    }    
//
//    outNet.run(image_debug);
//    cout <<"system pause" << endl; while(1);







//****************************** third stage ******************************
    count = 0;
    for(vector<struct Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            Mat thirdImage;
            resize(image(temp), thirdImage, Size(48, 48), 0, 0, cv::INTER_AREA);//INTER_LINEAR
            outNet.run(thirdImage);
            mydataFmt *pp=NULL;
            if(*(outNet.score_->pdata+1)>outNet.Othreshold){
                memcpy(it->regreCoord, outNet.location_->pdata, 4*sizeof(mydataFmt));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(outNet.score_->pdata+1);
//                pp = outNet.keyPoint_->pdata;
//                for(int num=0;num<5;num++){
//                    (it->ppoint)[num] = it->y1 + (it->y2 - it->y1)*(*(pp+num));
//                }
//                for(int num=0;num<5;num++){
//                    (it->ppoint)[num+5] = it->x1 + (it->x2 - it->x1)*(*(pp+num+5));
//                }
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

    if(count<1)return 2;
    refineAndSquareBbox(thirdBbox_, image.rows, image.cols);
    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
	int final_outBboxNum=0;
    for(vector<struct Bbox>::iterator it=thirdBbox_.begin(); it!=thirdBbox_.end();it++){
        if((*it).exist){
            rectangle(image, Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2), Scalar(0,0,255), 2,8,0);
			final_outBboxNum++;
//            for(int num=0;num<5;num++)circle(image,Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,Scalar(0,255,255), -1);
        }
    }
    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();
	printf("------Done find Face function\n");
	return final_outBboxNum;
}
