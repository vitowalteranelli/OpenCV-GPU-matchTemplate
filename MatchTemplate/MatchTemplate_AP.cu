#include "librerie.h" 
#include "MatchTemplate_AP.h"

NCVStatus ncvSquaredIntegralImage_device_AP(Ncv32f *d_src, Ncv32u srcStep, Ncv64f *d_dst, Ncv32u dstStep, NcvSize32u roi, INCVMemAllocator &gpuAllocator);

__device__ float normAcc(float num, float denum)
{
    if (::fabs(num) < denum)
        return num / denum;
    if (::fabs(num) < denum * 1.125f)
        return num > 0 ? 1 : -1;
    return 0;
}

int divUp(int a, int b)
{
	return (a + b - 1) / b;
}

template <int cn>
__global__ void normalizeKernel_32F_AP(int w, int h, const PtrStep<double> image_sqsum, double templ_sqsum, PtrStepSzf result)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < result.cols && y < result.rows)
    {
        float image_sqsum_ = (float)(
                (image_sqsum.ptr(y + h)[(x + w) * cn] - image_sqsum.ptr(y)[(x + w) * cn]) -
                (image_sqsum.ptr(y + h)[x * cn] - image_sqsum.ptr(y)[x * cn]));
        result.ptr(y)[x] = normAcc(result.ptr(y)[x], sqrtf(image_sqsum_ * templ_sqsum));
    }
}

void MatchTemplate_AP::normalize_32F_AP(int w, int h, const PtrStepSz<double> image_sqsum, double templ_sqsum, PtrStepSzf result, int cn, cudaStream_t stream)
{
    dim3 threads(32, 8);
    dim3 grid(divUp(result.cols, threads.x), divUp(result.rows, threads.y));

	normalizeKernel_32F_AP<1><<<grid, threads, 0, stream>>>(w, h, image_sqsum, templ_sqsum, result);
	
	cudaDeviceSynchronize();
}

NCVStatus nppiStSqrIntegralGetSize_32f64f_AP(NcvSize32u roiSize, Ncv32u *pBufsize, cudaDeviceProp &devProp)
{
    ncvAssertReturn(pBufsize != NULL, NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(roiSize.width > 0 && roiSize.height > 0, NPPST_INVALID_ROI);

    NCVMemStackAllocator gpuCounter(static_cast<Ncv32u>(devProp.textureAlignment));
    ncvAssertReturn(gpuCounter.isInitialized(), NPPST_MEM_INTERNAL_ERROR);

    NCVStatus ncvStat = ncvSquaredIntegralImage_device_AP(NULL, roiSize.width * sizeof(Ncv32f), NULL, (roiSize.width+1) * sizeof(Ncv64f), roiSize, gpuCounter);
    ncvAssertReturnNcvStat(ncvStat);

    *pBufsize = (Ncv32u)gpuCounter.maxSize();
    return NPPST_SUCCESS;
}

NCVStatus nppiStSqrIntegral_32f64f_C1R_AP(Ncv32f *d_src, Ncv32u srcStep, Ncv64f *d_dst, Ncv32u dstStep, NcvSize32u roiSize, Ncv8u *pBuffer, Ncv32u bufSize, cudaDeviceProp &devProp)
{
    NCVMemStackAllocator gpuAllocator(NCVMemoryTypeDevice, bufSize, static_cast<Ncv32u>(devProp.textureAlignment), pBuffer);
    ncvAssertReturn(gpuAllocator.isInitialized(), NPPST_MEM_INTERNAL_ERROR);

    NCVStatus ncvStat = ncvSquaredIntegralImage_device_AP(d_src, srcStep, d_dst, dstStep, roiSize, gpuAllocator);
    ncvAssertReturnNcvStat(ncvStat);

    return NPPST_SUCCESS;
}