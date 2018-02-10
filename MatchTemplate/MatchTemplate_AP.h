#include "librerie.h"

using namespace cv;
using namespace cv::gpu;
using namespace std;
using namespace device;

class MatchTemplate_AP
{

public:

	MatchTemplate_AP(void);
	~MatchTemplate_AP(void);

	void matchTemplate_AP(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method, Stream& stream = Stream::Null());

private:

	void matchTemplate_AP(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method, MatchTemplateBuf &buf, Stream& stream);
	void matchTemplate_CCORR_NORMED_32F_AP(const GpuMat& image, const GpuMat& templ, GpuMat& result, MatchTemplateBuf &buf, Stream& stream);
	void matchTemplate_CCORR_32F_AP(const GpuMat& image, const GpuMat& templ, GpuMat& result, MatchTemplateBuf &buf, Stream& stream);
	void matchTemplateNaive_CCORR_32F_AP(const PtrStepSzb image, const PtrStepSzb templ, PtrStepSzf result, int cn, cudaStream_t stream);
	void normalize_32F_AP(int w, int h, const PtrStepSz<double> image_sqsum, double templ_sqsum, PtrStepSzf result, int cn, cudaStream_t stream);
	void sqrIntegral_AP(const GpuMat& src, GpuMat& sqsum, Stream& s);
};

class NppStStreamHandler
{

public:

    inline explicit NppStStreamHandler(cudaStream_t newStream = 0)
    {
        oldStream = nppStSetActiveCUDAstream(newStream);
    }
    
    inline ~NppStStreamHandler()
    {
        nppStSetActiveCUDAstream(oldStream);
    }
    
private:

    cudaStream_t oldStream;
};