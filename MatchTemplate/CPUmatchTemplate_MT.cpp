#include "librerie.h"
#include "CPUmatchTemplate_MT.h"

CPUmatchTemplate_MT::CPUmatchTemplate_MT(void)
{
}

CPUmatchTemplate_MT::~CPUmatchTemplate_MT(void)
{
}

void CPUmatchTemplate_MT::match(Mat** master_mtx, Mat** slave_mtx, int rows, int cols, int factor, int inter_method, int tm_method)
{
	points = new Point*[rows];
	threads = new thread*[rows];
	
	HANDLE hthread[NUM_THREADS];

	for(int i = 0; i < rows; i++)
	{
		points[i] = new Point[cols];
		threads[i] = new thread[cols];
	}

	int max_threads_num = calcThreadNumber();
	
	int count = 0;

	cout << "    INFO: matching " << count << " di " << (rows * cols) << " completato\r";
	cout.flush();

	int k = 0;

	for(int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			threads[i][j] = thread([=](){matchThread(master_mtx, slave_mtx, rows, cols, factor, inter_method, tm_method, i, j); return 1;});
			
			hthread[k] = threads[i][j].native_handle();
			
			k++;

			count++;

			if (k == max_threads_num)
			{
				WaitForMultipleObjects(k, hthread, true, INFINITE);
				
				k = 0;
				
				cout << "    INFO: matching " << count << " di " << (rows * cols) << " completato\r";
				cout.flush(); 
			}
		}
	}

	cout << "    INFO: matching " << (rows * cols) << " di " << (rows * cols) << " completato\r";
	cout.flush();
	
	WaitForMultipleObjects(k, hthread, true, INFINITE);
	
	cout << endl;
}

void CPUmatchTemplate_MT::matchThread(Mat** master_mtx, Mat** slave_mtx, int rows, int cols, int factor, int inter_method, int tm_method, int i, int j)
{	
	//variabili locali
	Mat resized_master;
	Mat resized_slave;
	Mat result;

	int result_cols;
	int result_rows;

	double minVal; 
	double maxVal; 

	Point minLoc; 
	Point maxLoc;

	size_t clock_start;
	size_t clock_stop;
	
	clock_start = clock();
	resize(master_mtx[i][j], resized_master, Size(master_mtx[i][j].cols * factor, master_mtx[i][j].rows * factor), inter_method);
	resize(slave_mtx[i][j], resized_slave, Size(slave_mtx[i][j].cols * factor, slave_mtx[i][j].rows * factor), inter_method);
			
	result_cols = resized_master.cols - resized_slave.cols + 1;
	result_rows = resized_master.rows - resized_slave.rows + 1;

	result.create(result_cols, result_rows, CV_32FC1);
	
	matchTemplate(resized_master, resized_slave, result, tm_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1);
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
	clock_stop = clock();

	exec_time += double(clock_stop - clock_start) / CLOCKS_PER_SEC;
	
	points[i][j] = maxLoc;
	
	result.release();
	resized_slave.release();
	resized_master.release();
}

int CPUmatchTemplate_MT::calcThreadNumber()
{	
	MEMORYSTATUSEX statex;
	SYSTEM_INFO sysinfo;

	statex.dwLength = sizeof(statex);
	GlobalMemoryStatusEx(&statex);
	GetSystemInfo(&sysinfo);

	int CPU_num = sysinfo.dwNumberOfProcessors;

	unsigned master_resized_size = (unsigned)(MASTER_WIDTH * FACTOR) * (MASTER_HEIGHT * FACTOR) * DEEP;
	unsigned slave_resized_size = (unsigned)(SLAVE_WIDTH * FACTOR) * (SLAVE_HEIGHT * FACTOR) * DEEP;

	cout << "Master resized size: " << master_resized_size << endl;
	cout << "Slave resized size: " << slave_resized_size << endl;

	unsigned thread_size = sizeof(int) * 2 + sizeof(double) * 2 + sizeof(Point) * 2 + sizeof(size_t) * 2 + master_resized_size + slave_resized_size;
	(unsigned)thread_size /= 1024 * 8;

	cout << "Thread size: " << thread_size << endl;
	
	int max_allocable_threads = (unsigned)(statex.ullAvailPhys / 1024) / thread_size;
	
	unsigned max_concurrent_threads = std::thread::hardware_concurrency();

	int optimal_threads_num = (int)(statex.ullAvailPhys / 1024) * 2 / (thread_size * 3);
	
	cout << "    INFO: numero di core CPU: " << CPU_num << endl;
	cout << "    INFO: memoria disponibile (KB): " << statex.ullAvailPhys / 1024 << endl;
	cout << "    INFO: memoria occupata dal singolo thread (KB): " << thread_size << endl;
	cout << "    INFO: numero massimo di thread allocabili: " << max_allocable_threads << endl;
	cout << "    INFO: numero massimo di thread concorrenti: " << max_concurrent_threads << endl;
	
	if (max_allocable_threads > (int)max_concurrent_threads)
	{
		cout<< "    INFO: numero ottimale di thread allocabili: " << (int)max_concurrent_threads * 2 / 3 << endl << endl;

		return (int)max_concurrent_threads * 2 / 3;
	}
	else
	{
		cout<< "    INFO: numero ottimale di thread allocabili: " << optimal_threads_num << endl << endl;

		return optimal_threads_num;	
	}

	return optimal_threads_num;
}