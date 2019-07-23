//FPGA convolution author : Xing xiangrui
//Xiamen University  SmartDSP
//date 2018.11.1  14:54

#include "fpgaAcc.hpp"

//-----------------variables----------------------
//input size
int ImageCache::in_height;
int ImageCache::in_width;
int ImageCache::in_ChannelNum;
int ImageCache::in_ChannelPiexls;
int ImageCache::cur_IBRAM_addr;
int ImageCache::MAX_IBRAM_ADDR;
int MemoryController::in_width;
int MemoryController::stride;
int WeightsCache::outChannelNum;
int WeightsCache::inChannelNum;
//output size
int MemoryController::out_height;
int MemoryController::out_width;
int MemoryController::out_channelNum;
int MemoryController::out_channelPixels;
//memroy controller DRAM location
int MemoryController::peixl_out_DRAM_offset;//OBRAM to DRAM
int MemoryController::pixel_loadRow_DRAM_offset;//DRAM to IBRAM
int MemoryController::in_channel_pixels;

//current load location variable
int MemoryController::cur_loadPixel_row;
int MemoryController::cur_loadPiexel_col;
int MemoryController::load_pixel_offset;

//-------------Cache BRAM variable----------------
float OutputCache::OBRAM[MAX_NUM_CHOUT];
float ImageCache::IBRAM[MAX_IMAGE_CACHE_SIZE];
float WeightsCache::WBRAM[N_PE][MAX_2D_FILTERS_PER_PE][9];

//variable for test
int MAX_IBRAM_SIZE;
int MAXCO_PER_PE=0;
int MAX_FILTERS_PER_PE=0;

//----------------convolution in FPGA-----------------------------------
void convolution_3x3(int inHight,int inWidth,int inChanNum,int outHight,int outWidth,int OutChanNum,
					 int stride,
					 volatile float *SHARED_DRAM, int current_layer_offset){
#pragma HLS INTERFACE s_axilite register port=inHight bundle=axilite
#pragma HLS INTERFACE s_axilite register port=inWidth bundle=axilite
#pragma HLS INTERFACE s_axilite register port=inChanNum bundle=axilite
#pragma HLS INTERFACE s_axilite register port=outHight bundle=axilite
#pragma HLS INTERFACE s_axilite register port=outWidth bundle=axilite
#pragma HLS INTERFACE s_axilite register port=OutChanNum bundle=axilite
#pragma HLS INTERFACE s_axilite register port=stride bundle=axilite
#pragma HLS INTERFACE s_axilite register port=current_layer_offset bundle=axilite
#pragma HLS INTERFACE m_axi depth=DRAM_DEPTH port=SHARED_DRAM offset=slave bundle=memorybus

	printf("33conv");
	
	//pointer location initialize,
	//in each layer in order of weight, input, output
	volatile float * weight_ptr=&SHARED_DRAM[current_layer_offset];
	volatile float * input_ptr=&weight_ptr[9*inChanNum*OutChanNum];
	volatile float * output_ptr=&input_ptr[inHight*inWidth*inChanNum];

	//current varable for loop
	int cur_channel_out,cur_channel_in,cur_row_out,cur_col_out;
	int filter_col,filter_row;
	
	layer_setup:{
		MemoryController::setLayerConfig(inHight,inWidth,outHight,outWidth,OutChanNum,stride);
		ImageCache::setLayerConfig( inHight, inWidth, inChanNum);
		WeightsCache::setLayerConfig( inChanNum, OutChanNum);
	};
	
	//load weights from DRAM to WBRAM
	WeightsCache::load_WBRAM_from_DRAM(weight_ptr);
	
	MemoryController::setPixelLoadRowOffset();
	ImageCache::loadRowDRAM_2_IBRAM(input_ptr);
	MemoryController::setPixelLoadRowOffset();
	ImageCache::loadRowDRAM_2_IBRAM(input_ptr);
	
	//--------------nested loop for convolution----------------------------
	row_loop:for(cur_row_out=0;cur_row_out<MemoryController::out_height;cur_row_out++){//out_row,out_height
		//load row feature map into IBRAM
		while(MemoryController::cur_loadPixel_row < (cur_row_out*MemoryController::stride+3) ){
			MemoryController::setPixelLoadRowOffset();
			ImageCache::loadRowDRAM_2_IBRAM(input_ptr);
		}
	  
	  col_loop:for(cur_col_out=0;cur_col_out<MemoryController::out_width;cur_col_out++){//out_col,out_width
		MemoryController::setPixelOutOffset(cur_row_out,cur_col_out);
	    ch_in_loop:for(cur_channel_in=0;cur_channel_in<ImageCache::in_ChannelNum;cur_channel_in++){//in_channel
			
			//process input channel(process output channel)
			ProcessingElement::processInputChannel(cur_row_out*stride,cur_col_out*stride,
											cur_channel_in, MemoryController::out_channelNum);
		}//channel_in loop
		
		//write from accumulated OBRAM to DRAM
		ch_out_loop:for(cur_channel_out=0; cur_channel_out<MemoryController::out_channelNum; cur_channel_out++){
			MemoryController::writeBackOutputChannel(output_ptr,cur_channel_out, \
										OutputCache::OBRAM[cur_channel_out]);
		}
	 }//out_col loop
	}//out_row loop
};//--------------------------------convolution end-------------------------------


//--------------------Memory Controller----------------------
void MemoryController::setLayerConfig(int inHight,int inWidth,
									  int outHight,int outWidth,int OutChanNum,
									  int stride){
#pragma HLS inline
	MemoryController::out_height=outHight;
	MemoryController::out_width=outWidth;
	MemoryController::out_channelNum=OutChanNum;
	MemoryController::out_channelPixels=MemoryController::out_height*MemoryController::out_width;
	MemoryController::in_width=inWidth;
	MemoryController::stride=stride;
	MemoryController::in_channel_pixels=in_width*inHight;
	MemoryController::cur_loadPiexel_col=0;
	MemoryController::cur_loadPixel_row=0;
};
//write from OBRAM to DRAM
void MemoryController::writeBackOutputChannel(volatile float * output_ptr, int cur_co,
                                               float data) {
#pragma HLS inline
  int channel_offset=cur_co*out_channelPixels;
#pragma HLS RESOURCE variable = channel_offset core = MulnS latency = 2  
  output_ptr[channel_offset+peixl_out_DRAM_offset] = (volatile float) data;
};
//set output piexl offset on DRAM
void MemoryController::setPixelOutOffset(int cur_out_row,int cur_out_col){
#pragma HLS inline
  int row_offset=cur_out_row*out_width;
#pragma HLS RESOURCE variable = row_offset core = MulnS latency = 2  
  peixl_out_DRAM_offset=row_offset+cur_out_col;
};
//load input piexl from DRAM to IBRAM
void MemoryController::setPixelLoadRowOffset(){
#pragma HLS inline
//	int cur_inPixel_row_loc=cur_loadPixel_row*stride;
//#pragma HLS RESOURCE variable = cur_inPixel_row_loc core = MulnS latency = 2  	
	pixel_loadRow_DRAM_offset=cur_loadPixel_row*in_width;
#pragma HLS RESOURCE variable = pixel_loadRow_DRAM_offset core = MulnS latency = 2  
	cur_loadPiexel_col=0;
	cur_loadPixel_row++;
};
void MemoryController::setPixelLoadOffset(){
//#pragma HLS inline	
//	int cur_inPixel_col_loc=stride*cur_loadPiexel_col;
#pragma HLS RESOURCE variable = cur_inPixel_col_loc core = MulnS latency = 2  	
	load_pixel_offset=pixel_loadRow_DRAM_offset+cur_loadPiexel_col;
	cur_loadPiexel_col++;
};
//load from DRAM pixel to reg
float MemoryController::loadInputChannelPixel(volatile float * input_ptr,int ci){
#pragma HLS inline	
#pragma HLS pipeline
	int in_channel_pixel_offset=ci*in_channel_pixels;
#pragma HLS RESOURCE variable = in_channel_pixel_offset core = MulnS latency = 2  	
	float px=reg(input_ptr[load_pixel_offset+in_channel_pixel_offset]);
	return px;
};
//load from DRAM weight to reg
float MemoryController::load_weight_2_reg(volatile float * weight_DRAM_ptr, int weight_loc){
//#pragma HLS inline
//#pragma HLS pipeline
  float read = reg(weight_DRAM_ptr[weight_loc]);
  return read;
}

//------------------------ProcessingElement--------------------------
void ProcessingElement::macc2d(const float pixels[9],const float weights[9],
                               float& result) {
#pragma HLS inline
  float accumulator = 0.0f;
  float multresult[9];
#pragma HLS ARRAY_PARTITION variable = multresult complete dim = 0
L_MACC_multiply:
  for (int i = 0; i < 9; i++) {
#pragma HLS UNROLL
    multresult[i] = pixels[i] * weights[i];
  }
L_MACC_accumulate:
  for (int i = 0; i < 9; i++) {
#pragma HLS UNROLL
    accumulator = accumulator + multresult[i];
  }
  result = accumulator;
};
//get IBRAM pixel into buffer
void ProcessingElement::loadPixel_buffer(const int up_row,const int left_col,
							const int cur_In_channel, float pixel_buffer[9]){
#pragma HLS inline
#pragma HLS pipeline
  load_pixel_2_PE_row_loop:							
  for (int cur_filterRow=0;cur_filterRow<3;cur_filterRow++){
	int pixel_row_to_load=up_row+cur_filterRow;
	int IBRAM_line_offset=ImageCache::calcu_IBRAM_row_offset(pixel_row_to_load);
	load_pixel_2_PE_col_loop:
	for (int cur_filterCol=0;cur_filterCol<3;cur_filterCol++){
		int pixel_col_to_load=left_col+cur_filterCol;
		float px=reg(ImageCache::get_IBRAM_Pixel(IBRAM_line_offset,pixel_col_to_load,
								cur_In_channel));
		pixel_buffer[3*cur_filterRow+cur_filterCol]=px;
	}
  }
};

//load pixels[9] and loop weight on them
void ProcessingElement::processInputChannel(const int cur_row_times_stride,
											const int cur_col_times_stride,
											const int cur_ci, const int out_channelNum){
#pragma HLS inline off
#pragma HLS FUNCTION_INSTANTIATE variable = cur_ci
#pragma HLS dataflow	
	int cur_channel_in=cur_ci;
	float pixel_buffer[9];
#pragma HLS ARRAY_PARTITION variable = pixel_buffer complete dim = 0
	loadPixel_buffer(cur_row_times_stride, cur_col_times_stride,
						cur_channel_in, pixel_buffer);
	
	processAll_channelOut(out_channelNum, cur_channel_in,pixel_buffer);
};
//load and loop all channel out weight MACC on pixel[9]
void ProcessingElement::processAll_channelOut(const int out_Channel_Num, 
											  const int cur_ci,
											  const float pixel_buffer[9]){
#pragma HLS INLINE off
L_CH_OUT:for(int cur_co=0;cur_co<out_Channel_Num;cur_co++){
#pragma HLS unroll factor = N_PE
#pragma HLS PIPELINE
	float result,weights_local[9];
#pragma HLS ARRAY_PARTITION variable = weights_local complete dim = 0	
	// fetch weights
    WeightsCache::get_9_weights_to_buffer(cur_ci,cur_co,weights_local);
	//MACC 3*3  multiply accumulate
	macc2d(pixel_buffer,weights_local,result);
	//accumulate 3*3 macc result in OBRAM
	if (cur_ci == 0) {
		OutputCache::setOutChannel(cur_co, result);
	} else {
		OutputCache::accumulateChannel(cur_co, result);
	}
  };
};




//-------------------------WeightsCache-----------------------
void WeightsCache::setLayerConfig(int inChanNum,int OutChanNum){
#pragma HLS inline
	WeightsCache::outChannelNum=OutChanNum;
	WeightsCache::inChannelNum=inChanNum;
}

void WeightsCache::get_WBRAM_addr(const int cur_ci, const int cur_co,
									int &PEID, int &filterID){
#pragma HLS INLINE
	PEID=cur_co%N_PE;
	filterID=(cur_co/N_PE)*inChannelNum+cur_ci;
}
//load weights from DRAM to BRAM
void WeightsCache::load_WBRAM_from_DRAM(volatile float * weight_ptr){
#pragma HLS inline
	int PEID,filterID;
	float *WBRAM_ptr;volatile float *weight_DRAM_ptr;
	for(int cur_co=0;cur_co<outChannelNum;cur_co++){
	 int offset_inchannel=cur_co*inChannelNum;
#pragma HLS RESOURCE variable = offset_inchannel core = MulnS latency = 2	 
	 for(int cur_ci=0;cur_ci<inChannelNum;cur_ci++){
		get_WBRAM_addr(cur_ci,cur_co,PEID,filterID);
		WBRAM_ptr=WBRAM[PEID][filterID];
		int weight_DRAM_loc=9*(offset_inchannel+cur_ci);
#pragma HLS RESOURCE variable = weight_DRAM_loc core = MulnS latency = 2  
		weight_DRAM_ptr=weight_ptr+weight_DRAM_loc;
		for(int i=0;i<9;i++){
//#pragma HLS PIPELINE II = 2
#pragma HLS PIPELINE
			float weight_in_reg=MemoryController::load_weight_2_reg(weight_DRAM_ptr,i);
			WBRAM_ptr[i]=weight_in_reg;
		}
	 }
	}
}
//get 9 weights from IBRAM to buffer
void WeightsCache::get_9_weights_to_buffer(int cur_ci, int cur_co,float weight_buffer[9]){
#pragma HLS FUNCTION_INSTANTIATE variable = cur_co
#pragma HLS inline
#pragma HLS pipeline
// Array Partitioning
#pragma HLS ARRAY_PARTITION variable = WBRAM complete dim = 1    // PE ID
#pragma HLS ARRAY_PARTITION variable = WBRAM complete dim = 3    // weight ID
#pragma HLS RESOURCE variable = WBRAM core = RAM_S2P_BRAM latency = 3

#pragma HLS ARRAY_PARTITION variable = weight_buffer complete dim = 0
	int PEID,filterID;
	get_WBRAM_addr(cur_ci,cur_co,PEID,filterID);
	for(int i=0;i<9;i++){
		weight_buffer[i]=WBRAM[PEID][filterID][i];
	}
}



//-------------------------ImageCache-------------------------
void ImageCache::setLayerConfig(int inHight,int inWidth,int inChanNum){
#pragma HLS inline
	ImageCache::in_height=inHight;
	ImageCache::in_width=inWidth;
	ImageCache::in_ChannelNum=inChanNum;
	ImageCache::in_ChannelPiexls=ImageCache::in_width*ImageCache::in_height;
	ImageCache::MAX_IBRAM_ADDR = (in_width *in_ChannelNum* NUM_IMG_CACHE_LINES - 1);
	ImageCache::cur_IBRAM_addr=0;
}

//load whole row from DRAM to IBRAM (in hardware IBRAM order is row/col/channel_In)
void ImageCache::loadRowDRAM_2_IBRAM(volatile float * input_ptr){
#pragma HLS inline
	L_DRAM_PRELOADROW_X: for (int cur_col = 0; cur_col < in_width; cur_col++) {
		MemoryController::setPixelLoadOffset();
		loadPixelDRAM_2_IBRAM(input_ptr);
	}
};
void ImageCache::loadPixelDRAM_2_IBRAM(volatile float * input_ptr){
#pragma HLS inline
	L_PRELOAD_PIXEL_FROM_DRAM: for (int ci = 0; ci < in_ChannelNum; ci++) {
#pragma HLS pipeline
//#pragma HLS latency min=4
		float px = MemoryController::loadInputChannelPixel(input_ptr,ci);
		writeNextChannelPixel_2_IBRAM(px);
	}
};
void ImageCache::writeNextChannelPixel_2_IBRAM(float pixel){
	// Write Value into IBRAM
	IBRAM[cur_IBRAM_addr] = pixel;
	// Check and Wrap Write Address into IBRAM
	if (cur_IBRAM_addr == MAX_IBRAM_ADDR)
		cur_IBRAM_addr = 0;
	else
		cur_IBRAM_addr++;	
};

//load piexl from IBRAM out to PE
int ImageCache::calcu_IBRAM_row_offset(int cur_row){
#pragma HLS inline
	int IBRAM_line=cur_row%NUM_IMG_CACHE_LINES;
	int pixels_each_line=in_width*in_ChannelNum;
#pragma HLS RESOURCE variable=pixels_each_line core=MulnS latency=2	
	int IBRAM_line_offset=IBRAM_line*pixels_each_line;
#pragma HLS RESOURCE variable=IBRAM_line_offset core=MulnS latency=2	
	return IBRAM_line_offset;
};
float ImageCache::get_IBRAM_Pixel(const int IBRAM_line_offset, const int cur_col,
								const int channel_in){
#pragma HLS inline
#pragma HLS RESOURCE variable = IBRAM core = RAM_S2P_BRAM	
	int IBRAM_col_offset=cur_col*in_ChannelNum;
	int IBRAM_loc=IBRAM_line_offset+IBRAM_col_offset+channel_in;
	float px=IBRAM[IBRAM_loc];
	return px;
}




//------------------------OutputCache---------------------------
void OutputCache::accumulateChannel(int co, float value_to_add) {
#pragma HLS inline
#pragma HLS FUNCTION_INSTANTIATE variable = co
#pragma HLS ARRAY_PARTITION variable = OBRAM cyclic factor = N_PE
#pragma HLS RESOURCE variable=OBRAM core=RAM_T2P_BRAM latency=2
  float old_ch = getOutChannel(co); 
  float new_ch = old_ch + value_to_add;
  setOutChannel(co, new_ch); 
};

float OutputCache::getOutChannel(int co) {
#pragma HLS inline
  return OBRAM[co];
}

void OutputCache::setOutChannel(int co, float data) {
#pragma HLS inline
#pragma HLS FUNCTION_INSTANTIATE variable = co
  OBRAM[co] = data;
};




