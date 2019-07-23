//FPGA convolution author : Xing xiangrui
//Xiamen University  SmartDSP
//date 2018.11.1  14:54
#ifndef FPGAACC_H
#define FPGAACC_H

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <fstream>
#include <cstring>
//#include <cblas.h>
#include <string>
#include <math.h>
#include "pBox.h"

// ==========================
// = Architecture Constants =
// ==========================
// Number of Image Cache Lines (need 3, use 4 for simplified Addressing HW)
const int NUM_IMG_CACHE_LINES = 4;
// Number of Processing Elements
const int N_PE = 8;

const int DRAM_DEPTH = 5932576;

//IBRAM, OBRAM, WBRAM size (this value need to be changed by network)
const int MAX_IMAGE_CACHE_SIZE = 8704;
const int MAX_NUM_CHOUT = 128;
//const int MAX_CO_PER_PE=8;
// const int MAX_2D_FILTERS_PER_PE=1024;
const int MAX_2D_FILTERS_PER_PE= 2048;

void convolution_3x3(int inHight,int inWidth,int inChanNum,int outHight,int outWidth,int OutChanNum,
					 int stride,
					 volatile float *weight_ptr,volatile float *input_ptr,volatile float *output_ptr,
					 volatile float *weight_bias);


// Register Stage for manual Pipelining:
template <class T>  T reg(T x) {
#pragma HLS pipeline
#pragma HLS inline self off
#pragma HLS interface ap_ctrl_none register port=return
	return x;
}


namespace MemoryController {
	void setLayerConfig(int inHight,int inWidth,
						int outHight,int outWidth,int OutChanNum,
						int stride);
	float loadInputChannelPixel(volatile float  * input_ptr,int ci);
	void writeBackOutputChannel_test(volatile float * SHARED_DRAM, int co, float data);
	void writeBackOutputChannel(volatile float * SHARED_DRAM, int co, float data);
	void setPixelOutOffset(int cur_out_row,int cur_out_col);
	void setPixelLoadRowOffset();
	void setPixelLoadOffset();
	float load_weight_2_reg(volatile float * weight_DRAM_ptr, int weight_loc);
	
	//DRAM offset variable
	extern int peixl_out_DRAM_offset;
	extern int pixel_loadRow_DRAM_offset;
	extern int load_pixel_offset;

	//output size
	extern int out_height;
	extern int out_width;
	extern int out_channelNum;
	extern int out_channelPixels;
	
	//input size
	extern int in_width;
	extern int stride;
	
	//current variable
	extern int cur_loadPixel_row;
	extern int cur_loadPiexel_col;
	extern int in_channel_pixels;
};


namespace ProcessingElement{
	
	void macc2d(const float pixels[9],const float weights[9], float &result);
	void loadPixel_buffer(const int up_row,const int left_col,
						  const int cur_In_channel, float pixel_buffer[9]);
	void processAll_channelOut(const int out_Channel_Num, const int cur_ci,
							   const float pixel_buffer[9]);
	void processInputChannel(const int cur_row_times_stride,
							 const int cur_col_times_stride,
							 const int cur_ci, const int out_channelNum);
};

namespace OutputCache{
	
	void  accumulateChannel(int co, float value_to_add);
	float getOutChannel(int co);
	void  setOutChannel(int co, float data);
	
	extern float OBRAM[MAX_NUM_CHOUT];
	
};

namespace ImageCache{
	
	void setLayerConfig(int inHight,int inWidth,int inChanNum);
	
	void loadRowDRAM_2_IBRAM(volatile float * input_ptr);
	void loadPixelDRAM_2_IBRAM(volatile float * input_ptr);
	void writeNextChannelPixel_2_IBRAM(float pixel);
	int calcu_IBRAM_row_offset(int cur_row);
	float get_IBRAM_Pixel(const int IBRAM_line_offset, const int cur_col,
								const int channel_in);
	
	//input size
	extern int in_height;
	extern int in_width;
	extern int in_ChannelNum;
	extern int in_ChannelPiexls;
	
	//IBRAM address
	extern int MAX_IBRAM_ADDR;
	extern int cur_IBRAM_addr;
	
	extern float IBRAM[MAX_IMAGE_CACHE_SIZE];
};

namespace WeightsCache{
	
	void setLayerConfig(int inChanNum,int OutChanNum);
	void get_WBRAM_addr(const int cur_ci, const int cur_co,
									int &PEID, int &filterID);
	void load_WBRAM_from_DRAM(volatile float * weight_ptr);
	void get_9_weights_to_buffer(int cur_ci, int cur_co,float weight_buffer[9]);
	
	extern int outChannelNum;
	extern int inChannelNum;
	
	extern float WBRAM[N_PE][MAX_2D_FILTERS_PER_PE][9];
	
	
};

namespace BiasCache{
	void setLayerConfig(int OutChanNum);
    void load_BBRAM_from_DRAM(volatile float * weight_bias);
    extern int outChannelNum;
	extern float BBRAM[MAX_NUM_CHOUT];
};



#endif