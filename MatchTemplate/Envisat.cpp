#include "librerie.h"
#include "Envisat.h"

Envisat::Envisat(void)
{
	q_raster_layer = NULL;
	i_raster_layer = NULL;
	
	q_sample_addr = NULL;
	i_sample_addr = NULL;
    
	clip_dir = 0;
	
	q_band_name = "q";
	i_band_name = "i";
    
	subsampl = 1;
}

Envisat::~Envisat(void)
{
}

Mat Envisat::open(const char* file_name, String output_path)
{
	epr_init_api(e_log_debug, NULL, NULL);

    product_id = epr_open_product(file_name);
    
	if (product_id == NULL) 
	{
		cout << "*ERRORE: impossibile aprire il file " << file_name << endl << endl;
		system("pause");
		exit(-1);
	}

	width = (uint32)epr_get_scene_width(product_id);
    height = (uint32)epr_get_scene_height(product_id);

	//costruzione del nome del file di output    
	for (i = (int)strlen(file_name) - 1; i >= 0; i--)
	{
        if (strrchr("/\\", file_name[i]) != NULL)
		{
            clip_dir = 1;
            break;
        }
    }

    if (clip_dir) 
	{
        strncpy(out_name, file_name + i + 1, PATH_MAX);
        out_name[PATH_MAX] = '\0';
    } 
	else 
	{
        strcpy(out_name, file_name);
    }
	
    strcat(out_name, ".tif");

	//inizializzazione delle matrici
	img_32 = Mat(height, width, CV_32FC1);
	img_8 = Mat(height, width, CV_8UC1);

    //costruzione dei layer per le bande q e i
	q_raster_layer = make_layer(product_id, q_band_name, width, height, subsampl);
	i_raster_layer = make_layer(product_id, i_band_name, width, height, subsampl);

	for (row = 0; row < height; row++) 
	{
		for (col = 0; col < width; col++) 
		{
			q_sample_addr = (float*)q_raster_layer->buffer + q_raster_layer->raster_width * row + col;
			i_sample_addr = (float*)i_raster_layer->buffer + i_raster_layer->raster_width * row + col;
			
			//calcolo della magnitudine
			img_32.at<float>(row, col) = sqrt(pow(*q_sample_addr, 2) + pow(*i_sample_addr, 2));
		}
	}

	img_32.convertTo(img_8, CV_8UC1);
	imwrite(output_path + out_name, img_8);
	
	cout << "    INFO: altezza immagine raster (px): " << height << endl;
	cout << "    INFO: larghezza immagine raster (px): " << width << endl;
	cout << "    INFO: immagine " << out_name << " creata con successo" << endl;

	// releasing & closing
	img_8.release();
    epr_free_raster(q_raster_layer);
    epr_free_raster(i_raster_layer);
    epr_close_product(product_id);
    epr_close_api();

	return img_32;
}

//costruisce il raster del dataset specificato
EPR_SRaster* Envisat::make_layer(EPR_SProductId* product_id, const char* ds_name, uint32 source_w, uint32 source_h, uint32 subsampl)
{
    EPR_SBandId* band_id = NULL;
    int is_written;
    EPR_SRaster* raster_buffer = NULL;
    uint source_step_x, source_step_y;

    band_id = epr_get_band_id(product_id, ds_name);
    
	if (band_id == NULL) 
	{
		cout << "BAND_ID = NULL\n" << endl;
        return NULL;
    }

    source_step_x = subsampl;
    source_step_y = subsampl;

    raster_buffer = epr_create_compatible_raster(band_id, source_w, source_h, source_step_x, source_step_y);

    is_written = epr_read_band_raster(band_id, 0, 0, raster_buffer);

    if (is_written != 0) 
	{
        epr_free_raster(raster_buffer);
        return NULL;
    }

    return raster_buffer;
}