#include "librerie.h"

class CPUmatchTemplate_MT
{

public:

	Point** points;
	thread** threads;

	double exec_time;

	CPUmatchTemplate_MT(void);
	~CPUmatchTemplate_MT(void);

	void match(Mat** master_mtx, Mat** slave_mtx, int rows, int cols, int factor, int inter_method, int tm_method);

private:
	
	int calcThreadNumber(void);
	void matchThread(Mat** master_mtx, Mat** slave_mtx, int rows, int cols, int factor, int inter_method, int tm_method, int i, int j);
};