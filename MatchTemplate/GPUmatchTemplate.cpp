#include "librerie.h"
#include "GPUmatchTemplate.h"

GPUmatchTemplate::GPUmatchTemplate(void)
{
	matchTemplate_AP = MatchTemplate_AP();
}

GPUmatchTemplate::~GPUmatchTemplate(void)
{
}

void GPUmatchTemplate::match(Mat** master_mtx, Mat** slave_mtx, int rows, int cols, int factor, int inter_method, int tm_method)
{
	points = new Point*[rows];
		
	for (int i = 0; i < rows; i++)
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
			master.upload(master_mtx[i][j]);
			slave.upload(slave_mtx[i][j]);

			gpu::resize(master, resized_master, Size(master.cols * factor, master.rows * factor), inter_method);
			gpu::resize(slave, resized_slave, Size(slave.cols * factor, slave.rows * factor), inter_method);

			result_cols = resized_master.cols - resized_slave.cols + 1;
			result_rows = resized_master.rows - resized_slave.rows + 1;

			result.create(result_cols, result_rows, CV_32FC1);

			matchTemplate_AP.matchTemplate_AP(resized_master, resized_slave, result, tm_method);
			gpu::normalize(result, result, 0, 1, NORM_MINMAX, -1);
			gpu::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc); 
			clock_stop = clock();
			
			exec_time += double(clock_stop - clock_start) / CLOCKS_PER_SEC;

			count++;

			cout << "    INFO: matching " << count << " di " << (rows * cols) << " completato\r";
			cout.flush(); 
			
			points[i][j] = maxLoc;
			
			result.release();
			resized_slave.release();
			resized_master.release();
			slave.release();
			master.release();
		}
	}

	cout << endl;
}