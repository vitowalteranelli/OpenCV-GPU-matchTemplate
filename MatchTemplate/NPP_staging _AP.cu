#include "librerie.h" 
#include "MatchTemplate_AP.h"

texture<Ncv8u,  1, cudaReadModeElementType> tex8u;
//texture<Ncv32u, 1, cudaReadModeElementType> tex32u;
//texture<uint2,  1, cudaReadModeElementType> tex64u;

const Ncv32u NUM_SCAN_THREADS = 256;
const Ncv32u LOG2_NUM_SCAN_THREADS = 8;
NCV_CT_ASSERT(K_WARP_SIZE == 32);


template<class T_in, class T_out>
struct _scanElemOp
{
    template<bool tbDoSqr>
    static inline __host__ __device__ T_out scanElemOp(T_in elem)
    {
        return scanElemOp( elem, Int2Type<(int)tbDoSqr>() );
    }

private:

    template <int v> struct Int2Type { enum { value = v }; };

    static inline __host__ __device__ T_out scanElemOp(T_in elem, Int2Type<0>)
    {
        return (T_out)elem;
    }

    static inline __host__ __device__ T_out scanElemOp(T_in elem, Int2Type<1>)
    {
        return (T_out)(elem*elem);
    }
};


static Ncv32u getPaddedDimension(Ncv32u dim, Ncv32u elemTypeSize, Ncv32u allocatorAlignment)
{
    Ncv32u alignMask = allocatorAlignment-1;
    Ncv32u inverseAlignMask = ~alignMask;
    Ncv32u dimBytes = dim * elemTypeSize;
    Ncv32u pitch = (dimBytes + alignMask) & inverseAlignMask;
    Ncv32u PaddedDim = pitch / elemTypeSize;
    return PaddedDim;
}


template<class T>
inline __device__ T readElem(T *d_src, Ncv32u texOffs, Ncv32u srcStride, Ncv32u curElemOffs);

template<>
inline __device__ Ncv8u readElem<Ncv8u>(Ncv8u *d_src, Ncv32u texOffs, Ncv32u srcStride, Ncv32u curElemOffs)
{
    return tex1Dfetch(tex8u, texOffs + srcStride * blockIdx.x + curElemOffs);
}

template<>
inline __device__ Ncv32u readElem<Ncv32u>(Ncv32u *d_src, Ncv32u texOffs, Ncv32u srcStride, Ncv32u curElemOffs)
{
    return d_src[curElemOffs];
}

template<>
inline __device__ Ncv32f readElem<Ncv32f>(Ncv32f *d_src, Ncv32u texOffs, Ncv32u srcStride, Ncv32u curElemOffs)
{
    return d_src[curElemOffs];
}


template <class T>
inline __device__ T warpScanInclusive(T idata, volatile T *s_Data)
{
#if __CUDA_ARCH__ >= 300
    const unsigned int laneId = cv::gpu::device::Warp::laneId();

    // scan on shuffl functions
    #pragma unroll
    for (int i = 1; i <= (K_WARP_SIZE / 2); i *= 2)
    {
        const T n = cv::gpu::device::shfl_up(idata, i);
        if (laneId >= i)
              idata += n;
    }

    return idata;
#else
    Ncv32u pos = 2 * threadIdx.x - (threadIdx.x & (K_WARP_SIZE - 1));
    s_Data[pos] = 0;
    pos += K_WARP_SIZE;
    s_Data[pos] = idata;

    s_Data[pos] += s_Data[pos - 1];
    s_Data[pos] += s_Data[pos - 2];
    s_Data[pos] += s_Data[pos - 4];
    s_Data[pos] += s_Data[pos - 8];
    s_Data[pos] += s_Data[pos - 16];

    return s_Data[pos];
#endif
}


template <class T>
inline __device__ T warpScanExclusive(T idata, volatile T *s_Data)
{
    return warpScanInclusive(idata, s_Data) - idata;
}


template <class T, Ncv32u tiNumScanThreads>
inline __device__ T blockScanInclusive(T idata, volatile T *s_Data)
{
    if (tiNumScanThreads > K_WARP_SIZE)
    {
        //Bottom-level inclusive warp scan
        T warpResult = warpScanInclusive(idata, s_Data);

        //Save top elements of each warp for exclusive warp scan
        //sync to wait for warp scans to complete (because s_Data is being overwritten)
        __syncthreads();
        
		if( (threadIdx.x & (K_WARP_SIZE - 1)) == (K_WARP_SIZE - 1) )
        {
            s_Data[threadIdx.x >> K_LOG2_WARP_SIZE] = warpResult;
        }

        //wait for warp scans to complete
        __syncthreads();

        if( threadIdx.x < (tiNumScanThreads / K_WARP_SIZE) )
        {
            //grab top warp elements
            T val = s_Data[threadIdx.x];
            //calculate exclusive scan and write back to shared memory
            s_Data[threadIdx.x] = warpScanExclusive(val, s_Data);
        }

        //return updated warp scans with exclusive scan results
        __syncthreads();
        
		return warpResult + s_Data[threadIdx.x >> K_LOG2_WARP_SIZE];
    }
    else
    {
        return warpScanInclusive(idata, s_Data);
    }
}


template <class T_in, class T_out, bool tbDoSqr>
__global__ void scanRows(T_in *d_src, Ncv32u texOffs, Ncv32u srcWidth, Ncv32u srcStride, T_out *d_II, Ncv32u IIstride)
{
    //advance pointers to the current line
    if (sizeof(T_in) != 1)
    {
        d_src += srcStride * blockIdx.x;
    }
    
	//for initial image 8bit source we use texref tex8u
    d_II += IIstride * blockIdx.x;

    Ncv32u numBuckets = (srcWidth + NUM_SCAN_THREADS - 1) >> LOG2_NUM_SCAN_THREADS;
    Ncv32u offsetX = 0;

    __shared__ T_out shmem[NUM_SCAN_THREADS * 2];
    __shared__ T_out carryElem;
    carryElem = 0;
    __syncthreads();

    while (numBuckets--)
    {
        Ncv32u curElemOffs = offsetX + threadIdx.x;
        T_out curScanElem;

        T_in curElem;
        T_out curElemMod;

        if (curElemOffs < srcWidth)
        {
            //load elements
            curElem = readElem<T_in>(d_src, texOffs, srcStride, curElemOffs);
        }
        curElemMod = _scanElemOp<T_in, T_out>::scanElemOp<tbDoSqr>(curElem);

        //inclusive scan
        curScanElem = blockScanInclusive<T_out, NUM_SCAN_THREADS>(curElemMod, shmem);

        if (curElemOffs <= srcWidth)
        {
            //make scan exclusive and write the bucket to the output buffer
            d_II[curElemOffs] = carryElem + curScanElem - curElemMod;
            offsetX += NUM_SCAN_THREADS;
        }

        //remember last element for subsequent buckets adjustment
        __syncthreads();
        if (threadIdx.x == NUM_SCAN_THREADS-1)
        {
            carryElem += curScanElem;
        }
        __syncthreads();
    }

    if (offsetX == srcWidth && !threadIdx.x)
    {
        d_II[offsetX] = carryElem;
    }
}


template <bool tbDoSqr, class T_in, class T_out>
NCVStatus scanRowsWrapperDevice(T_in *d_src, Ncv32u srcStride, T_out *d_dst, Ncv32u dstStride, NcvSize32u roi)
{
    cudaChannelFormatDesc cfdTex;
    
	size_t alignmentOffset = 0;
    
	if (sizeof(T_in) == 1)
    {
        cfdTex = cudaCreateChannelDesc<Ncv8u>();
        ncvAssertCUDAReturn(cudaBindTexture(&alignmentOffset, tex8u, d_src, cfdTex, roi.height * srcStride), NPPST_TEXTURE_BIND_ERROR);
        if (alignmentOffset > 0)
        {
            ncvAssertCUDAReturn(cudaUnbindTexture(tex8u), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaBindTexture(&alignmentOffset, tex8u, d_src, cfdTex, alignmentOffset + roi.height * srcStride), NPPST_TEXTURE_BIND_ERROR);
        }
    }
    
	scanRows
        <T_in, T_out, tbDoSqr>
        <<<roi.height, NUM_SCAN_THREADS, 0, nppStGetActiveCUDAstream()>>>
        (d_src, (Ncv32u)alignmentOffset, roi.width, srcStride, d_dst, dstStride);

    ncvAssertCUDALastErrorReturn(NPPST_CUDA_KERNEL_EXECUTION_ERROR);

    return NPPST_SUCCESS;
}


NCVStatus ncvSquaredIntegralImage_device_AP(Ncv32f *d_src, Ncv32u srcStep, Ncv64f *d_dst, Ncv32u dstStep, NcvSize32u roi, INCVMemAllocator &gpuAllocator)
{
    ncvAssertReturn(gpuAllocator.isInitialized(), NPPST_MEM_INTERNAL_ERROR);
    ncvAssertReturn(gpuAllocator.memType() == NCVMemoryTypeDevice ||
                      gpuAllocator.memType() == NCVMemoryTypeNone, NPPST_MEM_RESIDENCE_ERROR);
    ncvAssertReturn((d_src != NULL && d_dst != NULL) || gpuAllocator.isCounting(), NPPST_NULL_POINTER_ERROR);
    ncvAssertReturn(roi.width > 0 && roi.height > 0, NPPST_INVALID_ROI);
    ncvAssertReturn(srcStep >= roi.width * sizeof(Ncv32f) &&
                    dstStep >= (roi.width + 1) * sizeof(Ncv64f) &&
					srcStep % sizeof(Ncv32f) == 0 &&
                    dstStep % sizeof(Ncv64f) == 0, NPPST_INVALID_STEP);

	srcStep /= sizeof(Ncv32f);
    dstStep /= sizeof(Ncv64f);

    Ncv32u WidthII = roi.width + 1;
    Ncv32u HeightII = roi.height + 1;
    Ncv32u PaddedWidthII32 = getPaddedDimension(WidthII, sizeof(Ncv32f), gpuAllocator.alignment());
    Ncv32u PaddedHeightII32 = getPaddedDimension(HeightII, sizeof(Ncv32f), gpuAllocator.alignment());
    Ncv32u PaddedWidthII64 = getPaddedDimension(WidthII, sizeof(Ncv64f), gpuAllocator.alignment());
    Ncv32u PaddedHeightII64 = getPaddedDimension(HeightII, sizeof(Ncv64f), gpuAllocator.alignment());
    Ncv32u PaddedWidthMax = PaddedWidthII32 > PaddedWidthII64 ? PaddedWidthII32 : PaddedWidthII64;
    Ncv32u PaddedHeightMax = PaddedHeightII32 > PaddedHeightII64 ? PaddedHeightII32 : PaddedHeightII64;

    NCVMatrixAlloc<Ncv32f> Tmp32_1(gpuAllocator, PaddedWidthII32, PaddedHeightII32);
    ncvAssertReturn(Tmp32_1.isMemAllocated(), NPPST_MEM_INTERNAL_ERROR);
    NCVMatrixAlloc<Ncv64f> Tmp64(gpuAllocator, PaddedWidthMax, PaddedHeightMax);
    ncvAssertReturn(Tmp64.isMemAllocated(), NPPST_MEM_INTERNAL_ERROR);

    NCVMatrixReuse<Ncv32f> Tmp32_2(Tmp64.getSegment(), gpuAllocator.alignment(), PaddedWidthII32, PaddedHeightII32);
    ncvAssertReturn(Tmp32_2.isMemReused(), NPPST_MEM_INTERNAL_ERROR);
    NCVMatrixReuse<Ncv64f> Tmp64_2(Tmp64.getSegment(), gpuAllocator.alignment(), PaddedWidthII64, PaddedHeightII64);
    ncvAssertReturn(Tmp64_2.isMemReused(), NPPST_MEM_INTERNAL_ERROR);

    NCVStatus ncvStat;
    NCV_SET_SKIP_COND(gpuAllocator.isCounting());

    NCV_SKIP_COND_BEGIN

    ncvStat = scanRowsWrapperDevice
		<true, Ncv32f, Ncv32f>
        (d_src, srcStep, Tmp32_2.ptr(), PaddedWidthII32, roi);
    ncvAssertReturnNcvStat(ncvStat);

    ncvStat = nppiStTranspose_32f_C1R(Tmp32_2.ptr(), PaddedWidthII32*sizeof(Ncv32f), Tmp32_1.ptr(), PaddedHeightII32*sizeof(Ncv32f), NcvSize32u(WidthII, roi.height));
    ncvAssertReturnNcvStat(ncvStat);

    ncvStat = scanRowsWrapperDevice
        <false, Ncv32f, Ncv64f>
        (Tmp32_1.ptr(), PaddedHeightII32, Tmp64_2.ptr(), PaddedHeightII64, NcvSize32u(roi.height, WidthII));
    ncvAssertReturnNcvStat(ncvStat);

    ncvStat = nppiStTranspose_64f_C1R(Tmp64_2.ptr(), PaddedHeightII64*sizeof(Ncv64f), d_dst, dstStep*sizeof(Ncv64f), NcvSize32u(HeightII, WidthII));
    ncvAssertReturnNcvStat(ncvStat);

    NCV_SKIP_COND_END

    return NPPST_SUCCESS;
}