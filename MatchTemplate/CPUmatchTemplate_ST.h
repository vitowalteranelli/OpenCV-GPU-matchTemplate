#include "librerie.h"

class CPUmatchTemplate_ST
{

public:

	Point** points;

	size_t clock_start;
	size_t clock_stop;

	double exec_time;

	CPUmatchTemplate_ST(void);
	~CPUmatchTemplate_ST(void);

	void match(Mat** master_mtx, Mat** slave_mtx, int rows, int cols, int factor, int inter_method, int tm_method);

private:

	Mat resized_master;
	Mat resized_slave;
	Mat result;

	int result_cols;
	int result_rows;

	double minVal; 
	double maxVal; 

	Point minLoc; 
	Point maxLoc;
};