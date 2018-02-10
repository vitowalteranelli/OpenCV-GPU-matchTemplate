#include "librerie.h"
#include "MatchTemplate_AP.h"																			

NCVStatus nppiStSqrIntegralGetSize_32f64f_AP(NcvSize32u roiSize, Ncv32u *pBufsize, cudaDeviceProp &devProp);

NCVStatus nppiStSqrIntegral_32f64f_C1R_AP(Ncv32f *d_src, Ncv32u srcStep, Ncv64f *d_dst, Ncv32u dstStep, NcvSize32u roiSize, Ncv8u *pBuffer, Ncv32u bufSize, cudaDeviceProp &devProp);

MatchTemplate_AP::MatchTemplate_AP(void)
{
}

MatchTemplate_AP::~MatchTemplate_AP(void)
{
}

void MatchTemplate_AP::matchTemplate_AP(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method, Stream& stream)
{
	MatchTemplateBuf buf;
    matchTemplate_AP(image, templ, result, method, buf, stream);
}

void MatchTemplate_AP::matchTemplate_AP(const GpuMat& image, const GpuMat& templ, GpuMat& result, int method, MatchTemplateBuf &buf, Stream& stream)
{
	matchTemplate_CCORR_NORMED_32F_AP(image, templ, result, buf, stream);
}

void MatchTemplate_AP::matchTemplate_CCORR_NORMED_32F_AP(const GpuMat& image, const GpuMat& templ, GpuMat& result, MatchTemplateBuf &buf, Stream& stream)
{
	matchTemplate_CCORR_32F_AP(image, templ, result, buf, stream);

	buf.image_sqsums.resize(1);
	sqrIntegral_AP(image.reshape(1), buf.image_sqsums[0], stream);

	double templ_sqsum = (double)sqrSum(templ.reshape(1))[0];
	normalize_32F_AP(templ.cols, templ.rows, buf.image_sqsums[0], templ_sqsum, result, image.channels(), StreamAccessor::getStream(stream));
}

void MatchTemplate_AP::matchTemplate_CCORR_32F_AP(const GpuMat& image, const GpuMat& templ, GpuMat& result, MatchTemplateBuf &buf, Stream& stream)
{
    result.create(image.rows - templ.rows + 1, image.cols - templ.cols + 1, CV_32FC1);

    ConvolveBuf convolve_buf;
    convolve_buf.user_block_size = buf.user_block_size;

    convolve(image.reshape(1), templ.reshape(1), result, true, convolve_buf, stream);
}

void MatchTemplate_AP::sqrIntegral_AP(const GpuMat& src, GpuMat& sqsum, Stream& s)
{
    NcvSize32u roiSize;
    roiSize.width = src.cols;
    roiSize.height = src.rows;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cv::gpu::getDevice());

    Ncv32u bufSize;
    nppiStSqrIntegralGetSize_32f64f_AP(roiSize, &bufSize, prop);
    GpuMat buf(1, bufSize, CV_32FC1);

    cudaStream_t stream = StreamAccessor::getStream(s);
	
    NppStStreamHandler h(stream);

    sqsum.create(src.rows + 1, src.cols + 1, CV_64FC1);
    nppiStSqrIntegral_32f64f_C1R_AP(const_cast<Ncv32f*>(src.ptr<Ncv32f>(0)), static_cast<int>(src.step), sqsum.ptr<Ncv64f>(0), static_cast<int>(sqsum.step), roiSize, buf.ptr<Ncv8u>(0), bufSize, prop);
	
    if (stream == 0)
		cudaDeviceSynchronize();
}