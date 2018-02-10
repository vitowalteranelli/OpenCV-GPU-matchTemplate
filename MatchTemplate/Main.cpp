#include "librerie.h"
#include "Envisat.h"
#include "CPUmatchTemplate_ST.h"
#include "CPUmatchTemplate_MT.h"
#include "GPUmatchTemplate.h"
	
//VARIABILI GLOBALI																	
Mat master;
Mat slave;
Mat centered_master;
Mat centered_slave;
Mat** master_mtx;
Mat** slave_mtx;

string master_path;
string slave_path;
string output_path = "output/";
string result_path = "risultati.txt";

CPUmatchTemplate_ST CPU_ST;
CPUmatchTemplate_MT CPU_MT;
GPUmatchTemplate GPU;
Envisat EnviHandler;

size_t clock_start;
size_t clock_stop;

double exec_time;

int cols;
int rows;

//DICHIARAZIONE DELLE FUNZIONI
void centerMasterSlave(void);
void splitMasterSlave(void);
void writeResults(void);
void cleanMemory(void);

int main()
{
	CPU_ST = CPUmatchTemplate_ST();
	CPU_MT = CPUmatchTemplate_MT();
	GPU = GPUmatchTemplate();
	EnviHandler = Envisat();

	system("mode con cols=100");
	system("mode con lines=50");

	 _COORD coord; 
    coord.X = 100; 
    coord.Y = 100; 

	HANDLE Handle = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleScreenBufferSize(Handle, coord);

	cout << endl;
	cout << setw(70) << "+++ ENVISAT IMAGES COREGISTRATION - First step +++" << endl << endl;
	cout << setw(62) << "Dott. ANELLI Vito Walter" << endl << endl;
	cout << setw(62) << "Dott. PAGLIARA Alessandro" << endl << endl << endl;

	//verifica le presenza di hardware CUDA
	try
	{
		//gpu::printCudaDeviceInfo(gpu::getDevice());
		gpu::resetDevice();
	}
	catch(exception e)
	{
		cout << "*ERRORE: nessun hardware CUDA disponibile." << endl << endl;
		system("pause");
		return 0;
	}
	
	cout << "- CARICAMENTO DATI INTERFEROMETRICI" << endl << endl;

	cout << " -> Input master: ";
	cin >> master_path;
	
	cout << endl;

	cout << " -> Input slave: ";
	cin >> slave_path;

	cout << endl;

	if ((GetFileAttributes((LPCWSTR)output_path.c_str())) == INVALID_FILE_ATTRIBUTES)
		_mkdir(output_path.c_str());

	cout << " -> Caricamento immagine master...Attendere" << endl << endl;
	clock_start = clock();
	master = EnviHandler.open(master_path.c_str(), output_path);
	clock_stop = clock();
	exec_time = double(clock_stop - clock_start) / CLOCKS_PER_SEC;
	cout << "    INFO: tempo di esecuzione (s): " << exec_time << endl << endl;
	
	cout << " -> Caricamento immagine slave...Attendere" << endl << endl;
	clock_start = clock();
	slave = EnviHandler.open(slave_path.c_str(), output_path);
	clock_stop = clock();
	exec_time = double(clock_stop - clock_start) / CLOCKS_PER_SEC;
	cout << "    INFO: tempo di esecuzione (s): " << exec_time << endl << endl;
		
	//allineamento master-slave
	cout << " -> Allineamento master-slave...Attendere" << endl << endl;
	clock_start = clock();
	centerMasterSlave();
	clock_stop = clock();
	exec_time = double(clock_stop - clock_start) / CLOCKS_PER_SEC;
	cout << "    INFO: altezza immagine allineata (px): " << centered_master.rows << endl;
	cout << "    INFO: larghezza immagine allineata (px): " << centered_master.cols << endl;
	cout << "    INFO: allineamento completato" << endl;
	cout << "    INFO: immagine centered_" << master_path << ".tif creata con successo" << endl;
	cout << "    INFO: immagine centered_" << slave_path << ".tif creata con successo" << endl;
	cout << "    INFO: tempo di esecuzione (s): " << exec_time << endl << endl;
	
	//splitting
	cout << " -> Splitting master-slave...Attendere" << endl << endl;
	clock_start = clock();
	splitMasterSlave();
	clock_stop = clock();
	exec_time = double(clock_stop - clock_start) / CLOCKS_PER_SEC;
	cout << "    INFO: splitting completato" << endl;
	cout << "    INFO: tempo di esecuzione (s): " << exec_time << endl << endl;
	
	//rilascio della memoria
	master.release();
	slave.release();
	centered_master.release();
	centered_slave.release();

	rows = cols = 10;
	
	cin.get();
	
	cout << "Premere un tasto per avviare il matching su CPU (single threading)...";
	cin.get();
	
	cout << endl;
	cout << "- MATCHING" << endl << endl;
	
	//CPUmatchTemplate
	cout << " -> CPU (single threading): Elaborazione in corso...Attendere" << endl << endl;
	clock_start = clock();
	CPU_ST.match(master_mtx, slave_mtx, rows, cols, FACTOR, INTER_METHOD, TM_METHOD);
	clock_stop = clock();
	exec_time = double(clock_stop - clock_start) / CLOCKS_PER_SEC;
	cout << "    INFO: tempo di esecuzione (s): " << CPU_ST.exec_time << endl;
	cout << "    INFO: tempo di esecuzione totale (s): " << exec_time << endl << endl;

	cout << "\a";

	cout << "Premere un tasto per avviare il matching su CPU (multithreading)...";
	cin.get();
	cout << endl;
	
	//CPUmatchTemplate-multithreaded
	cout << " -> CPU (multithreading): Elaborazione in corso...Attendere" << endl << endl;
	clock_start = clock();
	CPU_MT.match(master_mtx, slave_mtx, rows, cols, FACTOR, INTER_METHOD, TM_METHOD);
	clock_stop = clock();
	exec_time = double(clock_stop - clock_start) / CLOCKS_PER_SEC;
	cout << "    INFO: tempo di esecuzione (s): " << CPU_MT.exec_time << endl;
	cout << "    INFO: tempo di esecuzione totale (s): " << exec_time << endl << endl;

	//CPU_MT.exec_time è pari alla somma dei tempi di ogni singolo thread
	CPU_MT.exec_time = exec_time; 

	cout << "\a";
	
	cout << "Premere un tasto per avviare il matching su GPU...";
	cin.get();
	cout << endl;
	
	//GPUmatchTemplate
	cout << " -> GPU: Elaborazione in corso...Attendere" << endl << endl;
	clock_start = clock();
	GPU.match(master_mtx, slave_mtx, rows, cols, FACTOR, INTER_METHOD, TM_METHOD);
	clock_stop = clock();
	exec_time = double(clock_stop - clock_start) / CLOCKS_PER_SEC;
	cout << "    INFO: tempo di esecuzione (s): " << GPU.exec_time << endl;
	cout << "    INFO: tempo di esecuzione totale (s): " << exec_time << endl << endl;

	cout << "\a";
	
	cout << "Premere un tasto scrivere i risultati...";
	cin.get();
	cout << endl;

	//rilascio della GPU
	gpu::resetDevice();
	
	cout<<" -> Scrittura risultati in corso...Attendere" << endl << endl;
	clock_start = clock();
	writeResults();
	clock_stop = clock();
	exec_time = double(clock_stop - clock_start) / CLOCKS_PER_SEC;
	cout << "    INFO: " << result_path << " creato con successo" << endl;
	cout << "    INFO: scrittura completata" << endl;
	cout << "    INFO: tempo di esecuzione (s): " << exec_time << endl << endl;
	
	cout << "\a";
	
	cleanMemory();
	
	system("pause");
	return(0);
}

//DEFINIZIONE DELLE FUNZIONI

//allinea master e slave
void centerMasterSlave()
{
	double minVal; 
	double maxVal; 

	Point minLoc; 
	Point maxLoc;
	Point matchLoc;
	Point m0, m1, m2, m3;
	Point s0, s1, s2, s3;

	int x = (int)floor(slave.cols * 0.5) - 256;
	int y = (int)floor(slave.rows * 0.5) - 256;

	Rect rect(x, y, 512, 512);
	
	Mat templ = slave(rect);
	Mat result;

	int result_rows = master.rows - templ.rows + 1;
	int result_cols = master.cols - templ.cols + 1;

	result.create(result_cols, result_rows, CV_32FC1);

	matchTemplate(master, templ, result, TM_METHOD);
	normalize(result, result, 0, 1, NORM_MINMAX, -1);
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
	
	matchLoc = maxLoc;

	result.release();
	templ.release();

	if (matchLoc.x - x < 0)
		m0.x = 0;
	else
		m0.x = matchLoc.x - x;
	
	if (matchLoc.y - y < 0)
		m0.y = 0;
	else
		m0.y = matchLoc.y - y;

	m1.y = m0.y;

	if (matchLoc.x + x + 256 > master.cols)
		m1.x = master.cols;
	else
		m1.x = matchLoc.x + x + 256;

	m2.x = m1.x;

	if (matchLoc.y + y + 256 > master.rows)
		m2.y = master.rows;
	else
		m2.y = matchLoc.y + y + 256;

	m3.x = m0.x;
	m3.y = m2.y;

	s0.x = m0.x - matchLoc.x + x;
	s0.y = m0.y - matchLoc.y + y;

	s1.x = m1.x - matchLoc.x + x;
	s1.y = m1.y - matchLoc.y + y;

	s2.x = m2.x - matchLoc.x + x;
	s2.y = m2.y - matchLoc.y + y;

	s3.x = m3.x - matchLoc.x + x;
	s3.y = m3.y - matchLoc.y + y;

	Rect master_rect(m0.x, m0.y, m1.x - m0.x, m2.y - m1.y);
	Rect slave_rect(s0.x, s0.y, s1.x - s0.x, s2.y - s1.y);

	centered_master = master(master_rect);
	centered_slave = slave(slave_rect);

	centered_master.convertTo(master, CV_8UC1);
	centered_slave.convertTo(slave, CV_8UC1);

	imwrite(output_path + "centered_" + master_path + ".tif", master);
	imwrite(output_path + "centered_" + slave_path + ".tif", slave);
}

//splitta master e slave
void splitMasterSlave()
{
	int i, j, x, y;
	
	rows = (int)floor(centered_master.rows / MASTER_HEIGHT);
	cols = (int)floor(centered_master.cols / MASTER_WIDTH);

	//inizializzazione delle matrici
	master_mtx = new Mat*[rows];
	slave_mtx = new Mat*[rows];

	for(i = 0; i < rows; i++)
	{
		master_mtx[i] = new Mat[cols];
		slave_mtx[i] = new Mat[cols];
	}

	//splitting
	for (y = 0; y < centered_master.rows - MASTER_HEIGHT; y = y  + MASTER_HEIGHT)
	{
		for (x = 0; x < centered_master.cols - MASTER_WIDTH; x = x + MASTER_WIDTH)
		{
			Rect master_rect(x, y, MASTER_WIDTH, MASTER_HEIGHT);
			Rect slave_rect(x + OFFSET_X, y + OFFSET_Y, SLAVE_WIDTH, SLAVE_HEIGHT);
			
			j = (int)floor(x / MASTER_WIDTH);
			i = (int)floor(y / MASTER_HEIGHT);

			master_mtx[i][j] = centered_master(master_rect);
			slave_mtx[i][j] = centered_slave(slave_rect);
		}
	}
}

//scrive i risultati
void writeResults()
{
	ofstream output;
	output.open(result_path);

	output << "+++ COREGISTRAZIONE IMMAGINI ENVISAT +++" << endl << endl;

	output << "- Input master: " << master_path << endl;
	output << "- Input slave: " << slave_path << endl << endl;

	output << "- Matching eseguiti: " << rows * cols << endl << endl;

	output << "- Tempo CPU_ST: " << CPU_ST.exec_time << " s" << endl;
	output << "- Tempo CPU_MT: " << CPU_MT.exec_time << " s" << endl;
	output << "- Tempo GPU:    " << GPU.exec_time << " s" << endl << endl;

	output << "- Risultati:" << endl << endl;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			output << "CPU_ST: [" << i << "][" << j << "] = (" << CPU_ST.points[i][j].x << ", " << CPU_ST.points[i][j].y << ")" << endl;
			output << "CPU_MT: [" << i << "][" << j << "] = (" << CPU_MT.points[i][j].x << ", " << CPU_MT.points[i][j].y << ")" << endl;
			output << "GPU:    [" << i << "][" << j << "] = (" << GPU.points[i][j].x << ", " << GPU.points[i][j].y << ")" << endl << endl;
		}
	}

	output.close();
}

//pulisce la memoria
void cleanMemory()
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			master_mtx[i][j].release();
			slave_mtx[i][j].release();
		}
	}

	CPU_ST.~CPUmatchTemplate_ST();
	CPU_MT.~CPUmatchTemplate_MT();
	GPU.~GPUmatchTemplate();
	EnviHandler.~Envisat();
}
