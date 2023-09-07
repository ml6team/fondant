#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
// 2d
#define BLOCK_ROWS 16
#define BLOCK_COLS 16


template <typename T>
__device__ __forceinline__ unsigned char hasBit(T bitmap, unsigned char pos) {
    return (bitmap >> pos) & 1;
}


__device__ int32_t find(const int32_t *s_buf, int32_t n) {
    while (s_buf[n] != n)
        n = s_buf[n];
    return n;
}

__device__ int32_t find_n_compress(int32_t *s_buf, int32_t n) {
    const int32_t id = n;
    while (s_buf[n] != n) {
        n = s_buf[n];
        s_buf[id] = n;
    }
    return n;
}

__device__ void union_(int32_t *s_buf, int32_t a, int32_t b)
{
    bool done;
    do
    {
        a = find(s_buf, a);
        b = find(s_buf, b);

        if (a < b) {
            int32_t old = atomicMin(s_buf + b, a);
            done = (old == b);
            b = old;
        }
        else if (b < a){
            int32_t old = atomicMin(s_buf + a, b);
            done = (old == a);
            a = old;
        }
        else
            done = true;

    } while (!done);
}

namespace cc2d
{
    __global__ void init_labeling(int32_t *label, const uint32_t W, const uint32_t H)
    {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t idx = row * W + col;

        if (row < H && col < W)
            label[idx] = idx;
    }


    __global__ void merge(uint8_t *img, int32_t *label, const uint32_t W, const uint32_t H)
    {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t idx = row * W + col;

        if (row >= H || col >= W)
            return;

        uint32_t P = 0;

        // NOTE : Original Codes, but occurs silent error
        // NOTE : Programs keep runnig, but now showing printf logs, and the result is weird
        // uint8_t buffer[4] = {0};
        // if (col + 1 < W) {
        //     *(reinterpret_cast<uint16_t*>(buffer)) = *(reinterpret_cast<uint16_t*>(img + idx));
        //     if (row + 1 < H) {
        //         *(reinterpret_cast<uint16_t*>(buffer + 2)) = *(reinterpret_cast<uint16_t*>(img + idx + W));
        //     }
        // }
        // else {
        //     buffer[0] = img[idx];
        //     if (row + 1 < H)
        //         buffer[2] = img[idx + W];
        // }
        // if (buffer[0])              P |= 0x777;
        // if (buffer[1])              P |= (0x777 << 1);
        // if (buffer[2])              P |= (0x777 << 4);

        if (img[idx])                      P |= 0x777;
        if (row + 1 < H && img[idx + W])   P |= 0x777 << 4;
        if (col + 1 < W && img[idx + 1])   P |= 0x777 << 1;

        if (col == 0)               P &= 0xEEEE;
        if (col + 1 >= W)           P &= 0x3333;
        else if (col + 2 >= W)      P &= 0x7777;

        if (row == 0)               P &= 0xFFF0;
        if (row + 1 >= H)           P &= 0xFF;

        if (P > 0)
        {
            // If need check about top-left pixel(if flag the first bit) and hit the top-left pixel
            if (hasBit(P, 0) && img[idx - W - 1]){
                union_(label, idx, idx - 2 * W - 2); // top left block
            }

            if ((hasBit(P, 1) && img[idx - W]) || (hasBit(P, 2) && img[idx - W + 1]))
                union_(label, idx, idx - 2 * W); // top bottom block

            if (hasBit(P, 3) && img[idx + 2 - W])
                union_(label, idx, idx - 2 * W + 2); // top right block

            if ((hasBit(P, 4) && img[idx - 1]) || (hasBit(P, 8) && img[idx + W - 1]))
                union_(label, idx, idx - 2); // just left block
        }
    }

    __global__ void compression(int32_t *label, const int32_t W, const int32_t H)
    {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t idx = row * W + col;

        if (row < H && col < W)
            find_n_compress(label, idx);
    }

    __global__ void final_labeling(const uint8_t *img, int32_t *label,
//                                   thrust::device_vector<int32_t> *pVec,
                                   const int32_t W, const int32_t H)
    {
        const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t idx = row * W + col;

        if (row >= H || col >= W)
            return;

        int32_t y = label[idx] + 1;

        if (img[idx])
            label[idx] = y;
        else
            label[idx] = 0;

        if (col + 1 < W)
        {
            if (img[idx + 1])
                label[idx + 1] = y;
            else
                label[idx + 1] = 0;

            if (row + 1 < H)
            {
                if (img[idx + W + 1])
                    label[idx + W + 1] = y;
                else
                    label[idx + W + 1] = 0;
            }
        }

        if (row + 1 < H)
        {
            if (img[idx + W])
                label[idx + W] = y;
            else
                label[idx + W] = 0;
        }
    }

} // namespace cc2d

torch::Tensor connected_componnets_labeling_2d(const torch::Tensor &input) {
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(input.ndimension() == 2, "input must be a  [H, W] shape");
    AT_ASSERTM(input.scalar_type() == torch::kUInt8, "input must be a uint8 type");

    const uint32_t H = input.size(-2);
    const uint32_t W = input.size(-1);

    AT_ASSERTM((H % 2) == 0, "shape must be a even number");
    AT_ASSERTM((W % 2) == 0, "shape must be a even number");

    // label must be uint32_t
    auto label_options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    torch::Tensor label = torch::zeros({H, W}, label_options);

    dim3 grid = dim3(((W + 1) / 2 + BLOCK_COLS - 1) / BLOCK_COLS, ((H + 1) / 2 + BLOCK_ROWS - 1) / BLOCK_ROWS);
    dim3 block = dim3(BLOCK_COLS, BLOCK_ROWS);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cc2d::init_labeling<<<grid, block, 0, stream>>>(
        label.data_ptr<int32_t>(), W, H
    );
    cc2d::merge<<<grid, block, 0, stream>>>(
        input.data_ptr<uint8_t>(),
        label.data_ptr<int32_t>(),
        W, H
    );
    cc2d::compression<<<grid, block, 0, stream>>>(
        label.data_ptr<int32_t>(), W, H
    );
    cc2d::final_labeling<<<grid, block, 0, stream>>>(
        input.data_ptr<uint8_t>(),
        label.data_ptr<int32_t>(),
        W, H
    );

    return label;
}

torch::Tensor connected_componnets_labeling_2d_batch(const torch::Tensor &input) {
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(input.ndimension() == 3, "input must be a [C, H, W] shape");
    AT_ASSERTM(input.scalar_type() == torch::kUInt8, "input must be a uint8 type");

    const uint32_t H = input.size(-2);
    const uint32_t W = input.size(-1);
    const uint32_t B = input.size(-3);

    AT_ASSERTM((H % 2) == 0, "shape must be a even number");
    AT_ASSERTM((W % 2) == 0, "shape must be a even number");

    // label must be uint32_t
    auto label_options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    torch::Tensor label = torch::zeros({B, H, W}, label_options);

    dim3 grid = dim3(((W + 1) / 2 + BLOCK_COLS - 1) / BLOCK_COLS, ((H + 1) / 2 + BLOCK_ROWS - 1) / BLOCK_ROWS);
    dim3 block = dim3(BLOCK_COLS, BLOCK_ROWS);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    for (int i = 0; i < B; ++i) {
        cc2d::init_labeling<<<grid, block, 0, stream>>>(
            label[i].data_ptr<int32_t>(), W, H
        );
        cc2d::merge<<<grid, block, 0, stream>>>(
            input[i].data_ptr<uint8_t>(),
            label[i].data_ptr<int32_t>(),
            W, H
        );
        cc2d::compression<<<grid, block, 0, stream>>>(
            label[i].data_ptr<int32_t>(), W, H
        );
        cc2d::final_labeling<<<grid, block, 0, stream>>>(
            input[i].data_ptr<uint8_t>(),
            label[i].data_ptr<int32_t>(),
            W, H
        );
    }
    return label;
}

