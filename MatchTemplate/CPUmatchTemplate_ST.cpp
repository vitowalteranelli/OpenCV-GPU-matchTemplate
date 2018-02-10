#include "librerie.h"
#include "CPUmatchTemplate_ST.h"

CPUmatchTemplate_ST::CPUmatchTemplate_ST(void)
{
}

CPUmatchTemplate_ST::~CPUmatchTemplate_ST(void)
{
}

void CPUmatchTemplate_ST::match(Mat** master_mtx, Mat** slave_mtx, int rows, int cols, int factor, int inter_method, int tm_method)
{
	points = new Point*[rows];

	for(int i = 0; i < rows; i++)
	{
		points[i] = new Point[cols];
	}

	int count = 0;

	cout << "    INFO: matching " << count << " di " << (rows * cols) << " completato\r";
	cout.flush(); 

	for(int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			clock_start = clock();
			resize(master_mtx[i][j], resized_master, Size(master_mtx[i][j].cols * factor, master_mtx[i][j].rows * factor), inter_method);
			resize(slave_mtx[i][j], resized_slave, Size(slave_mtx[i][j].cols * factor, slave_mtx[i][j].rows * factor), inter_method);
			
			result_cols = resized_master.cols - resized_slave.cols + 1;
			result_rows = resized_master.rows - resized_slave.rows + 1;

			result.create(result_cols, result_rows, CV_32FC1);
			
			matchTemplate(resized_master, resized_slave, result, tm_method);
			normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
			minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
			clock_stop = clock();
			
			exec_time += double(clock_stop - clock_start) / CLOCKS_PER_SEC;
			
			count++;

			cout << "    INFO: matching " << count << " di " << (rows * cols) << " completato\r";
			cout.flush(); 

			points[i][j] = maxLoc;

			result.release();
			resized_slave.release();
			resized_master.release();
		}
	}

	cout << endl;
}