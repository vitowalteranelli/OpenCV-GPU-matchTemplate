#include "librerie.h"

class Envisat
{

public:

	Envisat(void);
	~Envisat(void);

	Mat open(const char* file_name, String output_path);

private:

	EPR_SProductId* product_id;
	EPR_SRaster* q_raster_layer;
	EPR_SRaster* i_raster_layer;
	
	float* q_sample_addr;
	float* i_sample_addr;
	
	uint32 row;
    uint32 col;
    uint32 width;
    uint32 height;
	uint32 subsampl;
    
	char out_name[PATH_MAX + 1];
    
	int i;
	int clip_dir;
	
	double minVal;
	double maxVal;
    
	const char* q_band_name;
	const char* i_band_name;
   
	Mat img_32;
	Mat img_8;

	EPR_SRaster* make_layer(EPR_SProductId* product_id, const char* ds_name, uint32 source_w, uint32 source_h, uint32 subsampl);
};