#include "librerie.h"
#include "MatchTemplate_AP.h"

class GPUmatchTemplate
{

public:

	Point** points;

	size_t clock_start;
	size_t clock_stop;

	double exec_time;

	GPUmatchTemplate(void);
	~GPUmatchTemplate(void);

	void match(Mat** master_mtx, Mat** slave_mtx, int rows, int cols, int factor, int inter_method, int tm_method);

private:

	gpu::GpuMat master;
	gpu::GpuMat slave;
	gpu::GpuMat result;
	
	gpu::GpuMat resized_master;
	gpu::GpuMat resized_slave;

	int result_cols;
	int result_rows;

	double minVal; 
	double maxVal; 

	Point minLoc; 
	Point maxLoc;

	MatchTemplate_AP matchTemplate_AP;
};