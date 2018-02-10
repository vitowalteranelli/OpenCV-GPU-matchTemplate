#pragma warning( disable : 4273 )

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <opencv2\gpu\stream_accessor.hpp>
#include <opencv2\gpu\NCV.hpp>
#include <opencv2\gpu\NPP_staging.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>

#include <chrono>
#include <vector>
#include <windows.h>
#include <direct.h>
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <thread>

#include "epr_api.h"
#include "epr_string.h"
#include "tiffio.h"

#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;
using namespace cv;

//numero di threads
#define NUM_THREADS 100

//profondità in bit dei dati
#define DEEP 32

//lunghezza massima del nome dell'output
#define PATH_MAX 1023

//dimensioni immagine master
#define MASTER_WIDTH 256
#define MASTER_HEIGHT 256

//dimensioni immagine slave
#define SLAVE_WIDTH 128
#define SLAVE_HEIGHT 128

//offset slave
#define OFFSET_X 63
#define OFFSET_Y 63

//fattore di sovracampionamento
#define FACTOR 32

//metodo di interpolazione (3 = interpolazione bicubica)
#define INTER_METHOD 3

//metodo per il template matching (3 = crosscorrelazione normalizzata)
#define TM_METHOD 3