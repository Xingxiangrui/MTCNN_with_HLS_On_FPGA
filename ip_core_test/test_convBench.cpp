#include <time.h>
#include <stdio.h>
#include "fpgaAcc.hpp"

void convolution(const Weight *weightIn, const pBox *pboxIn, pBox *outpBox);

int main()
{
printf("Test Start SUCCESS!\n");//-------------------------------
	
	//conv parameters
	int inputSize=23; int inChannelNum=32;
	int OutChannelNum=64;
	int kernelSize=3; int Stride=2;
	int outputSize=((inputSize-kernelSize)/Stride)+1;
	int Input_Pixels=inputSize*inputSize*inChannelNum;
	int Output_Pixels=outputSize*outputSize*OutChannelNum;
	int weightkernel_Pixels=9*inChannelNum*OutChannelNum;
	
	//conv variable
	Weight weightIn;
	pBox featureIn;
	pBox conv_PL_out;
	pBox conv_PS_out;
	
	//initialize conv weight variable
	weightIn.out_ChannelNum=OutChannelNum;
	weightIn.in_ChannelNum=inChannelNum;
	weightIn.kernelSize=kernelSize;
	weightIn.stride=Stride;
	
	//initialize buffer for current layer
	volatile float * current_layer_ptr=(volatile float *)malloc(sizeof(float)*(Input_Pixels+Output_Pixels+weightkernel_Pixels));

	weightIn.pdata=&current_layer_ptr[0];
	featureIn.pdata=&weightIn.pdata[weightkernel_Pixels];
	conv_PL_out.pdata=&featureIn.pdata[Input_Pixels];
	
	//weightIn.pdata=(volatile float *)malloc(sizeof(float)*weightkernel_Pixels);

	for (int i=0;i<weightkernel_Pixels;i++){
		weightIn.pdata[i]=(rand()%100)/100.0;
	}

	//initialize conv Input variable
	featureIn.width=inputSize;
	featureIn.height=inputSize;
	featureIn.channel=inChannelNum;
	//featureIn.pdata=(volatile float*)malloc(sizeof(float)*Input_Pixels);
	for (int i=0;i<Input_Pixels;i++){
		featureIn.pdata[i]=(rand()%100)/100.0;
	}
	
	//initialize conv Output variable
	conv_PL_out.width=outputSize;
	conv_PL_out.height=outputSize;
	conv_PL_out.channel=OutChannelNum;
	conv_PS_out.width=outputSize;
	conv_PS_out.height=outputSize;
	conv_PS_out.channel=OutChannelNum;	
	conv_PS_out.pdata=(volatile float*)malloc(sizeof(float)*Output_Pixels);
	//conv_PL_out.pdata=(volatile float*)malloc(sizeof(float)*Output_Pixels);
	memset((void *)conv_PS_out.pdata,0,sizeof(float)*Output_Pixels);
	memset((void *)conv_PL_out.pdata,0,sizeof(float)*Output_Pixels);
	
	//print memory location
	printf("Weight  memory location 0x%8x \n",weightIn.pdata);
	printf("Feature memory location 0x%8x \n",featureIn.pdata);
	printf("Output  memory location 0x%8x \n",conv_PL_out.pdata);
	
printf("Variable init SUCCESS!\n");//---------------------------------------
	
	//conv in PS
	convolution(&weightIn,&featureIn,&conv_PS_out);
	
printf("Conv in PS SUCCESS!\n");//--------------------------------
	
	//conv in PL
	convolution_3x3(featureIn.height, featureIn.width ,featureIn.channel,
						 conv_PL_out.height,conv_PL_out.width,conv_PL_out.channel,
						 weightIn.stride,
						 current_layer_ptr, 0);
printf("Conv in PL SUCCESS!\n");//--------------------------------
	
	//compare in PS and PL
	int error=0;
	for(int i=0;i<Output_Pixels;i++){
		if(conv_PS_out.pdata[i]!=conv_PL_out.pdata[i]){
			printf("Convolution ERROR!\n");
			printf("i is %d, value in PS= %f, in PL is %f\n",i,conv_PS_out.pdata[i],conv_PL_out.pdata[i]);
			error=1;
		}
	}
	printf("Compare DONE SUCCESS!\n");
	if(error==0)
		printf("PS and PL conv match SUCCESS!\n");
	else
		printf("PS and PL conv match FAILURE!\n");
	
/* 	free((void *)weightIn.pdata);
	free((void *)featureIn.pdata);
	free((void *)conv_PS_out.pdata);
	free((void *)conv_PL_out.pdata); */
	
	
    return 0;
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
	volatile float* weight_ptr=weightIn->pdata;volatile float *input_ptr=pboxIn->pdata;volatile float *output_ptr=outpBox->pdata;
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
