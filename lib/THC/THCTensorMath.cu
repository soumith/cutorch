#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

struct TensorFillOp {
  TensorFillOp(float v) : val(v) {}
  __device__ __forceinline__ void operator()(float* v) { *v = val; }

  const float val;
};

void THCudaTensor_fill(THCState* state, THCudaTensor *self_, float value)
{
  if (!THCudaTensor_pointwiseApply1(state, self_, TensorFillOp(value))) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_zero(THCState *state, THCudaTensor *self_)
{
  if (THCudaTensor_isContiguous(state, self_)) {
    THCudaCheck(cudaMemsetAsync(THCudaTensor_data(state, self_),
                                0,
                                sizeof(float) * THCudaTensor_nElement(state, self_)));
  } else {
    if (!THCudaTensor_pointwiseApply1(state, self_, TensorFillOp(0))) {
      THArgCheck(false, 1, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_zeros(THCState *state, THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_zero(state, r_);
}

void THCudaTensor_ones(THCState *state, THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_fill(state, r_, 1);
}

void THCudaTensor_reshape(THCState *state, THCudaTensor *r_, THCudaTensor *t, THLongStorage *size)
{
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_copy(state, r_, t);
}

long THCudaTensor_numel(THCState *state, THCudaTensor *t)
{
  return THCudaTensor_nElement(state, t);
}

struct TensorAddConstantOp {
  TensorAddConstantOp(float v) : val(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = *in + val;
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v += val;
  }

  const float val;
};

void THCudaTensor_add(THCState *state, THCudaTensor *self_, THCudaTensor *src_, float value)
{
  if (self_ == src_) {
    if (!THCudaTensor_pointwiseApply1(state, self_, TensorAddConstantOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src_);

    if (!THCudaTensor_pointwiseApply2(state, self_, src_, TensorAddConstantOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorMulConstantOp {
  TensorMulConstantOp(float v) : val(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = *in * val;
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v *= val;
  }

  const float val;
};

void THCudaTensor_mul(THCState *state, THCudaTensor *self_, THCudaTensor *src_, float value)
{
  if (self_ == src_) {
    if (!THCudaTensor_pointwiseApply1(state, self_, TensorMulConstantOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src_);

    if (!THCudaTensor_pointwiseApply2(state, self_, src_, TensorMulConstantOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_div(THCState* state, THCudaTensor *self_, THCudaTensor *src_, float value)
{
  THArgCheck(value != 0.0f, 3, "divide by zero");

  if (self_ == src_) {
    if (!THCudaTensor_pointwiseApply1(state, self_, TensorMulConstantOp(1.0f / value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src_);

    if (!THCudaTensor_pointwiseApply2(state, self_, src_, TensorMulConstantOp(1.0f / value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorAddOp {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out += *in;
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = *in1 + *in2;
  }
};

struct TensorCAddOp {
  TensorCAddOp(float v) : val(v) {}

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out += val * *in;
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = *in1 + val * *in2;
  }

  float val;
};

void THCudaTensor_cadd(THCState *state, THCudaTensor *self_, THCudaTensor* src1, float value, THCudaTensor *src2)
{
  THArgCheck(THCudaTensor_nElement(state, src1) ==
             THCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    if (value == 1.0f) {
      // self += src2
      if (!THCudaTensor_pointwiseApply2(state, self_, src2, TensorAddOp())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self += value * src2
      if (!THCudaTensor_pointwiseApply2(state, self_, src2, TensorCAddOp(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src1);

    if (value == 1.0f) {
      // self = src1 + src2
      if (!THCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorAddOp())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self = src1 + value * src2
      if (!THCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorCAddOp(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorMulOp {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out *= *in;
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = *in1 * *in2;
  }
};

void THCudaTensor_cmul(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THArgCheck(THCudaTensor_nElement(state, src1) ==
             THCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self *= src2
    if (!THCudaTensor_pointwiseApply2(state, self_, src2, TensorMulOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src1);

    // self = src1 * src2
    if (!THCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorMulOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorCPowOp {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = powf(*out, *in);
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = powf(*in1, *in2);
  }
};

void THCudaTensor_cpow(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THArgCheck(THCudaTensor_nElement(state, src1) ==
             THCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self = pow(self, src2)
    if (!THCudaTensor_pointwiseApply2(state, self_, src2, TensorCPowOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src1);

    // self = pow(src1, src2)
    if (!THCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorCPowOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorDivOp {
  __device__ __forceinline__ void
  operator()(float* out, float* in) {
    *out /= *in;
  }

  __device__ __forceinline__ void
  operator()(float* out, float* in1, float* in2) {
    *out = *in1 / *in2;
  }
};

void THCudaTensor_cdiv(THCState* state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THArgCheck(THCudaTensor_nElement(state, src1) ==
             THCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self *= src2
    if (!THCudaTensor_pointwiseApply2(state, self_, src2, TensorDivOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src1);

    // self = src1 * src2
    if (!THCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorDivOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorAddCMulOp {
  TensorAddCMulOp(float v) : val(v) {}

  __device__ __forceinline__ void
  operator()(float* out, float* in1, float* in2) {
    *out += val * *in1 * *in2;
  }

  float val;
};

void THCudaTensor_addcmul(THCState *state, THCudaTensor *self_, THCudaTensor *t, float value, THCudaTensor *src1, THCudaTensor *src2)
{
  if(self_ != t)
  {
    THCudaTensor_resizeAs(state, self_, t);
    THCudaTensor_copy(state, self_, t);
  }
  THCudaTensor_resizeAs(state, self_, src1);

  THArgCheck(THCudaTensor_nElement(state, src1) ==
             THCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (!THCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorAddCMulOp(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorAddCDivOp {
  TensorAddCDivOp(float v) : val(v) {}

  __device__ __forceinline__ void
  operator()(float* out, float* in1, float* in2) {
    *out += val * *in1 / *in2;
  }

  float val;
};

void THCudaTensor_addcdiv(THCState *state, THCudaTensor *self_, THCudaTensor *t, float value, THCudaTensor *src1, THCudaTensor *src2)
{
  if(self_ != t)
  {
    THCudaTensor_resizeAs(state, self_, t);
    THCudaTensor_copy(state, self_, t);
  }

  THCudaTensor_resizeAs(state, self_, src1);
  THArgCheck(THCudaTensor_nElement(state, src1) == THCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (!THCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorAddCDivOp(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

float THCudaTensor_dot(THCState *state, THCudaTensor *self, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(state, self) == THCudaTensor_nElement(state, src), 2, "sizes do not match");

  {
    self = THCudaTensor_newContiguous(state, self);
    src = THCudaTensor_newContiguous(state, src);

    float result = THCudaBlas_dot(state,
                                  THCudaTensor_nElement(state, self),
                                  THCudaTensor_data(state, self), 1,
                                  THCudaTensor_data(state, src), 1);
    THCudaTensor_free(state, src);
    THCudaTensor_free(state, self);

    return result;
  }
}

float THCudaTensor_minall(THCState *state, THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(state, self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(state, self), (float)(THInf), thrust::minimum<float>());

  THCudaTensor_free(state, self);
  return result;
}

float THCudaTensor_maxall(THCState *state, THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(state, self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(state, self), (float)(-THInf), thrust::maximum<float>());

  THCudaTensor_free(state, self);
  return result;
}

float THCudaTensor_sumall(THCState *state, THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(state, self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(state, self), (float)(0), thrust::plus<float>());

  THCudaTensor_free(state, self);
  return result;
}

float THCudaTensor_prodall(THCState *state, THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(state, self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(state, self), (float)(1), thrust::multiplies<float>());

  THCudaTensor_free(state, self);
  return result;
}

struct dim4 {
    unsigned arr[4];

    __host__ dim4(unsigned init=0) {
        for(unsigned i=0; i<4; i++) { arr[i] = init; }
    }

    __host__ __device__ unsigned& operator[](const unsigned& idx) { return arr[idx]; }
};

void THCudaTensor_sum(THCState* state, THCudaTensor *self, THCudaTensor *src, long dimension)
{
  THCudaTensor_reduceDim(
    state, self, src,
    thrust::identity<float>(), thrust::plus<float>(), 0.0f, dimension);

  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_prod(THCState* state, THCudaTensor *self, THCudaTensor *src, long dimension)
{
  THCudaTensor_reduceDim(
    state, self, src,
    thrust::identity<float>(), thrust::multiplies<float>(), 1.0f, dimension);

  THCudaCheck(cudaGetLastError());
}

/* Perform an inclusive scan along an outer dimension of a tensor.
 *
 * - num_orows is the size of the flattened outer dimensions;
 * - num_irows is the size of the flattened inner dimensions;
 * - row_size is the size of the dimension along which to compute the variance;
 *
 * The dimensions to the outside and inside of the specified dimension are considered as flattened.
 * Thread blocks with the same blockIdx.y process an "outer row" (i.e. an element of the flattened
 * outer dimensions, which contains several "inner rows").
 * Each thread processes a single inner row at a time.
 */
template<class BinaryOp>
__global__ void THCudaTensor_kernel_scanOuterDim(float *tgt_, float *src_,
                                                 unsigned num_orows, unsigned num_irows, unsigned row_size,
                                                 float init, BinaryOp binary_op)
{
  for (unsigned orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (unsigned irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      float *src = src_ + orow * row_size * num_irows + irow;
      float *tgt = tgt_ + orow * row_size * num_irows + irow;
      float acc = init;

      for (unsigned col = 0; col < row_size; ++col) {
        acc = binary_op(acc, *src);
        *tgt = acc;

        src += num_irows;
        tgt += num_irows;
      }
    }
  }
}

template<class BinaryOp>
__host__ void THCudaTensor_scanOuterDim(THCState *state, THCudaTensor *tgt, THCudaTensor *src, long dimension,
                                        float init, BinaryOp binary_op)
{
  unsigned ndim = THCudaTensor_nDimension(state, src);
  // Treat all outer dimensions (i.e. dim < dimension) as one.
  unsigned num_orows = 1;
  for (unsigned dim = 0; dim < dimension; dim++) {
    num_orows *= THCudaTensor_size(state, src, dim);
  }
  unsigned row_size = THCudaTensor_size(state, src, dimension);
  // Treat all inner dimensions (i.e. dim > dimension) as one.
  unsigned num_irows = 1;
  for (unsigned dim = dimension + 1; dim < ndim; dim++) {
    num_irows *= THCudaTensor_size(state, src, dim);
  }

  dim3 threads(min(512, num_irows));
  unsigned maxGridDim = 1024;
  dim3 grid(min(maxGridDim, num_orows), min(maxGridDim, DIVUP(num_irows, threads.x)));

  THCudaTensor_kernel_scanOuterDim<<<grid, threads>>>(
      THCudaTensor_data(state, tgt), THCudaTensor_data(state, src), num_orows, num_irows, row_size, init, binary_op);
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}


/* Perform an inclusive scan along the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<int num_threads_x, int num_threads_y, class BinaryFunction>
__global__ void THCudaTensor_kernel_scanInnermostDim(float *tgt_, float *src_,
                                                     unsigned num_rows, unsigned row_size,
                                                     float init, BinaryFunction binary_op)
{
  __shared__ float sbuf[num_threads_y][2 * num_threads_x];

  float* row_buf = sbuf[threadIdx.y];

  for (unsigned block_row = blockIdx.x * blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    unsigned row = block_row + threadIdx.y;
    float block_total = init;

    float *row_src = src_ + row * row_size;
    float *row_tgt = tgt_ + row * row_size;

    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (unsigned block_col = 0; block_col < row_size; block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      unsigned col1 = block_col + threadIdx.x;
      unsigned col2 = block_col + num_threads_x + threadIdx.x;
      if (row < num_rows) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = row_src[col1];
        } else {
          row_buf[threadIdx.x] = init;
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] = row_src[col2];
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
        }

        // Add the total value of all previous blocks to the first value of this block.
        if (threadIdx.x == 0) {
          row_buf[0] = binary_op(row_buf[0], block_total);
        }
      }
      __syncthreads();

      // Parallel reduction (up-sweep).
      for (unsigned s = num_threads_x, d = 1; s >= 1; s >>= 1, d <<= 1) {
        if (row < num_rows && threadIdx.x < s) {
          unsigned offset = (2 * threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      // Down-sweep.
      for (unsigned s = 2, d = num_threads_x / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < num_rows && threadIdx.x < s - 1) {
          unsigned offset = 2 * (threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      // Write back to output.
      if (row < num_rows) {
        if (col1 < row_size) row_tgt[col1] = row_buf[threadIdx.x];
        if (col2 < row_size) row_tgt[col2] = row_buf[num_threads_x + threadIdx.x];
      }
      block_total = row_buf[2 * num_threads_x - 1];
      __syncthreads();
    }
  }
}

template<class BinaryFunction>
__host__ void THCudaTensor_scanInnermostDim(THCState *state, THCudaTensor *tgt, THCudaTensor *src, float init, BinaryFunction binary_op)
{
  unsigned ndim = THCudaTensor_nDimension(state, src);
  // Treat all outer dimensions as a single dimension.
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THCudaTensor_size(state, src, dim);
  }
  unsigned row_size = THCudaTensor_size(state, src, ndim - 1);

  dim3 threads(16, 32);
  dim3 grid(min(1024, DIVUP(num_rows, threads.y)));

  THCudaTensor_kernel_scanInnermostDim<16, 32><<<grid, threads>>>(
      THCudaTensor_data(state, tgt), THCudaTensor_data(state, src), num_rows, row_size, init, binary_op);
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

template<class BinaryFunction>
void THCudaTensor_scanDim(THCState *state, THCudaTensor *self_, THCudaTensor *src, long dimension, float init, BinaryFunction binary_op)
{
  THCudaTensor_resizeAs(state, self_, src);

  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  src = THCudaTensor_newContiguous(state, src);

  if (dimension == THCudaTensor_nDimension(state, src) - 1) {
    THCudaTensor_scanInnermostDim(state, self, src, init, binary_op);
  } else {
    THCudaTensor_scanOuterDim(state, self, src, dimension, init, binary_op);
  }

  THCudaTensor_free(state, src);
  THCudaTensor_freeCopyTo(state, self, self_);
}

void THCudaTensor_cumsum(THCState *state, THCudaTensor *self, THCudaTensor *src, long dimension)
{
  return THCudaTensor_scanDim(state, self, src, dimension, 0.0f, thrust::plus<float>());
}

void THCudaTensor_cumprod(THCState *state, THCudaTensor *self, THCudaTensor *src, long dimension)
{
  return THCudaTensor_scanDim(state, self, src, dimension, 1.0f, thrust::multiplies<float>());
}

/* A set of reduction kernels that take in binary ops on thrust pairs (of value, index).
   These are useful when you not only have to do a reduction, but you might have
   to preserve the location of contention (for example min/max operations).
   The structure of the kernels follows the structure of the reduction kernels.
*/
template<class BinaryFunction>
__global__ void THCudaTensor_kernel_transformReduceOuterDimIndex(float *tgt1, float *tgt2,
                                                             float *src_,
                                                             unsigned num_orows,
                                                             unsigned num_irows,
                                                             unsigned row_size,
                                                             thrust::pair<float,float> init,
                                                             BinaryFunction binary_op)
{
  for (unsigned orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (unsigned irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      float *src = src_ + orow * row_size * num_irows + irow;
      thrust::pair<float,float> acc = init;

      for (unsigned col = 0; col < row_size; ++col) {
        acc = binary_op(thrust::make_pair(*src, col+1), acc); // i+1 for 1-indexing
        src += num_irows;
      }
      tgt1[orow * num_irows + irow] = acc.first;
      tgt2[orow * num_irows + irow] = acc.second;
    }
  }
}

template<class BinaryFunction>
__host__ void THCudaTensor_transformReduceOuterDimIndex(THCState *state, THCudaTensor *tgt1, THCudaTensor *tgt2,
                                                   THCudaTensor *src,
                                                   long rdim, thrust::pair<float,float> init,
                                                   BinaryFunction binary_op)
{
  unsigned ndim = THCudaTensor_nDimension(state, src);
  unsigned num_orows = 1;
  for (unsigned dim = 0; dim < rdim; dim++) {
    num_orows *= THCudaTensor_size(state, src, dim);
  }
  unsigned row_size = THCudaTensor_size(state, src, rdim);
  unsigned num_irows = 1;
  for (unsigned dim = rdim + 1; dim < ndim; dim++) {
    num_irows *= THCudaTensor_size(state, src, dim);
  }

  dim3 threads(min(512, num_irows));
  unsigned maxGridDim = 1024;
  dim3 grid(min(maxGridDim, num_orows), min(maxGridDim, DIVUP(num_irows, threads.x)));

  THCudaTensor_kernel_transformReduceOuterDimIndex<<<grid, threads>>>(
    THCudaTensor_data(state, tgt1), THCudaTensor_data(state, tgt2),
    THCudaTensor_data(state, src), num_orows, num_irows, row_size, init, binary_op);
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

/* Reduce the innermost dimension of a tensor (on thrust::pair functors which are (value, index))
 *
 * For an n-d tensor (n <= 4) where the reduction is along the innermost dimension:
 *
 * - block.x is the innermost dimension, i.e. dimension 0;
 * - block.y and grid.y make up dimension 1; and
 * - grid.x and grid z are the remaining two outer dimensions (if any)
 *
 * Reduction along other dimensions is handled in a separate kernel.
 */
template<class BinaryFunction>
__global__ void THCudaTensor_kernel_transformReduceInnermostDimIndex(
  float *tgt1, float* tgt2, float *src_,
  unsigned num_rows, unsigned row_size,
  thrust::pair<float,float> init, BinaryFunction binary_op)
{
  __shared__ float sbuf[32][16];
  __shared__ float ibuf[32][16];

  for (unsigned block_row = blockIdx.x * blockDim.y; block_row < num_rows; block_row += blockDim.y * gridDim.x) {
    unsigned row = block_row + threadIdx.y;
    thrust::pair<float,float> acc = init;
    if (row < num_rows) {
      float *src = src_ + row * row_size;
      // Sequential reduction within a thread.
      for (unsigned col = threadIdx.x; col < row_size; col += blockDim.x) {
        acc = binary_op(thrust::make_pair(src[col], col+1), acc);
      }
    }

    sbuf[threadIdx.y][threadIdx.x] = acc.first;
    ibuf[threadIdx.y][threadIdx.x] = acc.second;

    // Reduce intermediate values to single value.
    float* sline = &sbuf[threadIdx.y][0];
    float* iline = &ibuf[threadIdx.y][0];
    for (unsigned s = 8; s > 0; s >>= 1) {
      if (row < num_rows && threadIdx.x < s) {
        thrust::pair<float,float> arg1 = thrust::make_pair<float,float>(sline[threadIdx.x], iline[threadIdx.x]);
        thrust::pair<float,float> arg2 = thrust::make_pair<float,float>(sline[threadIdx.x + s], iline[threadIdx.x + s]);
        thrust::pair<float,float> res = binary_op(arg1, arg2);
        sline[threadIdx.x] = res.first;
        iline[threadIdx.x] = res.second;
      }
      __syncthreads();
    }

    if (row < num_rows && threadIdx.x == 0) {
      tgt1[row] = sline[0];
      tgt2[row] = iline[0];
    }
    __syncthreads();
  }
}

template<class BinaryFunction>
__host__ void THCudaTensor_transformReduceInnermostDimIndex(
  THCState *state, THCudaTensor *tgt1, THCudaTensor *tgt2, THCudaTensor *src,
  thrust::pair<float,float> init, BinaryFunction binary_op)
{
  unsigned ndim = THCudaTensor_nDimension(state, src);
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THCudaTensor_size(state, src, dim);
  }
  unsigned row_size = THCudaTensor_size(state, src, ndim - 1);

  dim3 threads(16, 32);
  dim3 grid(min(1024, DIVUP(num_rows, threads.y)));

  THCudaTensor_kernel_transformReduceInnermostDimIndex<<<grid, threads>>>(
    THCudaTensor_data(state, tgt1), THCudaTensor_data(state, tgt2),
    THCudaTensor_data(state, src), num_rows, row_size, init, binary_op);
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

template<class BinaryFunction>
void THCudaTensor_reduceDimIndex(THCState *state, THCudaTensor *tgt1_, THCudaTensor *tgt2_, THCudaTensor *src,
                             long dimension, thrust::pair<float,float> init,
                                     BinaryFunction binary_op)
{
  THArgCheck(dimension >= 0 && dimension < THCudaTensor_nDimension(state, src), 3, "dimension out of range");

  THLongStorage *dim = THCudaTensor_newSizeOf(state, src);
  THLongStorage_set(dim, dimension, 1);
  THCudaTensor_resize(state, tgt1_, dim, NULL);
  THCudaTensor_resize(state, tgt2_, dim, NULL);
  THLongStorage_free(dim);

  THCudaTensor *tgt1 = THCudaTensor_newContiguous(state, tgt1_);
  THCudaTensor *tgt2 = THCudaTensor_newContiguous(state, tgt2_);
  src = THCudaTensor_newContiguous(state, src);

  if(dimension == THCudaTensor_nDimension(state, src)-1) {
    THCudaTensor_transformReduceInnermostDimIndex(state, tgt1, tgt2, src, init, binary_op);
  } else {
    THCudaTensor_transformReduceOuterDimIndex(state, tgt1, tgt2, src, dimension, init, binary_op);
  }

  THCudaTensor_free(state, src);
  THCudaTensor_freeCopyTo(state, tgt1, tgt1_);
  THCudaTensor_freeCopyTo(state, tgt2, tgt2_);
}

struct maxvalue_functor
{
  __host__ __device__ thrust::pair<float,float> operator()(const thrust::pair<float,float> &a,
                                                            const thrust::pair<float,float> &b)
  {
    if (a.first > b.first) return a;
    else return b;
  }
};

void THCudaTensor_max(THCState *state, THCudaTensor *values, THCudaTensor *indices, THCudaTensor *src, long dimension)
{
  const float minfloat32 = -3.402823466e+38f;
  thrust::pair<float,float> init = thrust::make_pair<float,float>(minfloat32, -1);
  return THCudaTensor_reduceDimIndex(state, values, indices, src, dimension, init,
                                 maxvalue_functor());
}

struct minvalue_functor
{
  __host__ __device__ thrust::pair<float,float> operator()(const thrust::pair<float,float> &a,
                                                            const thrust::pair<float,float> &b)
  {
    if (a.first < b.first) return a;
    else return b;
  }
};

void THCudaTensor_min(THCState *state, THCudaTensor *values, THCudaTensor *indices, THCudaTensor *src, long dimension)
{
  const float maxfloat32 = 3.402823466e+38f;
  thrust::pair<float,float> init = thrust::make_pair<float,float>(maxfloat32, -1);
  return THCudaTensor_reduceDimIndex(state, values, indices, src, dimension, init,
                                     minvalue_functor());
}


void THCudaTensor_addmv(THCState *state, THCudaTensor *r_, float beta, THCudaTensor *t, float alpha, THCudaTensor *mat, THCudaTensor *vec)
{
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected");

  if( mat->size[1] != vec->size[0] )
    THError("size mismatch");

  if(t->nDimension != 1)
    THError("size mismatch");

  if(t->size[0] != mat->size[0])
    THError("size mismatch");

  if(r_ != t)
  {
    THCudaTensor_resizeAs(state, r_, t);
    THCudaTensor_copy(state, r_, t);
  }

  if(mat->stride[0] == 1)
  {
    THCudaBlas_gemv(state, 'n', mat->size[0], mat->size[1],
                    alpha, THCudaTensor_data(state, mat), mat->stride[1],
                    THCudaTensor_data(state, vec), vec->stride[0],
                    beta, THCudaTensor_data(state, r_), r_->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    THCudaBlas_gemv(state, 't',  mat->size[1], mat->size[0],
                    alpha, THCudaTensor_data(state, mat), mat->stride[0],
                    THCudaTensor_data(state, vec), vec->stride[0],
                    beta, THCudaTensor_data(state, r_), r_->stride[0]);
  }
  else
  {
    THCudaTensor *cmat = THCudaTensor_newContiguous(state, mat);

    THCudaBlas_gemv(state, 't',  mat->size[1], mat->size[0],
                    alpha, THCudaTensor_data(state, cmat), cmat->stride[0],
                    THCudaTensor_data(state, vec), vec->stride[0],
                    beta, THCudaTensor_data(state, r_), r_->stride[0]);

    THCudaTensor_free(state, cmat);
  }
}

void THCudaTensor_addmm(THCState *state, THCudaTensor *r_, float beta, THCudaTensor *t, float alpha, THCudaTensor *m1, THCudaTensor *m2)
{
  char transpose_r, transpose_m1, transpose_m2;
  THCudaTensor *r__, *m1_, *m2_;

  if( (m1->nDimension != 2) || (m2->nDimension != 2) )
    THError("matrix and matrix expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) || (m1->size[1] != m2->size[0]) )
    THError("size mismatch");

  if(t != r_)
  {
    THCudaTensor_resizeAs(state, r_, t);
    THCudaTensor_copy(state, r_, t);
  }

  /* r_ */
  if(r_->stride[0] == 1)
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride[1] == 1)
  {
    THCudaTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';

    r__ = THCudaTensor_newWithSize2d(state, r_->size[1], r_->size[0]);
    THCudaTensor_copy(state, r__, r_);
    THCudaTensor_transpose(state, r__, NULL, 0, 1);
  }

  /* m1 */
  if(m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THCudaTensor_newContiguous(state, m1);
  }

  /* m2 */
  if(m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THCudaTensor_newContiguous(state, m2);
  }

  /* do the operation */
  THCudaBlas_gemm(state,
                  transpose_m1,
                  transpose_m2,
                  r__->size[(transpose_r == 'n' ? 0 : 1)],
                  r__->size[(transpose_r == 'n' ? 1 : 0)],
                  m1_->size[(transpose_r == 'n' ? 1 : 0)],
                  alpha,
                  THCudaTensor_data(state, m1_),
                  (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                  THCudaTensor_data(state, m2_),
                  (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                  beta,
                  THCudaTensor_data(state, r__),
                  r__->stride[(transpose_r == 'n' ? 1 : 0)]);

  /* free intermediate variables */
  if(m1_ != m1)
    THCudaTensor_free(state, m1_);

  if(m2_ != m2)
    THCudaTensor_free(state, m2_);

  if(r__ != r_)
    THCudaTensor_freeCopyTo(state, r__, r_);
}

void THCudaTensor_addr(THCState *state, THCudaTensor *r_, float beta, THCudaTensor *t, float alpha, THCudaTensor *vec1, THCudaTensor *vec2)
{
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) )
    THError("size mismatch");

  if(r_ != t)
  {
    THCudaTensor_resizeAs(state, r_, t);
    THCudaTensor_copy(state, r_, t);
  }

  if(beta != 1)
    THCudaTensor_mul(state, r_, r_, beta);

  if(r_->stride[0] == 1)
  {
    THCudaBlas_ger(state, vec1->size[0], vec2->size[0],
                   alpha, THCudaTensor_data(state, vec1), vec1->stride[0],
                   THCudaTensor_data(state, vec2), vec2->stride[0],
                   THCudaTensor_data(state, r_), r_->stride[1]);
  }
  else if(r_->stride[1] == 1)
  {
    THCudaBlas_ger(state, vec2->size[0], vec1->size[0],
                   alpha, THCudaTensor_data(state, vec2), vec2->stride[0],
                   THCudaTensor_data(state, vec1), vec1->stride[0],
                   THCudaTensor_data(state, r_), r_->stride[0]);
  }
  else
  {
    THCudaTensor *cr = THCudaTensor_newClone(state, r_);

    THCudaBlas_ger(state, vec2->size[0], vec1->size[0],
                   alpha, THCudaTensor_data(state, vec2), vec2->stride[0],
                   THCudaTensor_data(state, vec1), vec1->stride[0],
                   THCudaTensor_data(state, cr), cr->stride[0]);

    THCudaTensor_freeCopyTo(state, cr, r_);
  }
}

void THCudaTensor_baddbmm(THCState *state, THCudaTensor *result, float beta, THCudaTensor *t,
                          float alpha, THCudaTensor *batch1, THCudaTensor *batch2) {
  THArgCheck(THCudaTensor_nDimension(state, t) == 3, 4, "expected 3D tensor");
  THArgCheck(THCudaTensor_nDimension(state, batch1) == 3, 6, "expected 3D tensor");
  THArgCheck(THCudaTensor_nDimension(state, batch2) == 3, 7, "expected 3D tensor");
  THArgCheck(THCudaTensor_size(state, t, 0) == THCudaTensor_size(state, batch1, 0), 6,
             "equal number of batches expected");
  THArgCheck(THCudaTensor_size(state, t, 0) == THCudaTensor_size(state, batch2, 0), 7,
             "equal number of batches expected");
  THArgCheck(THCudaTensor_size(state, t, 1) == THCudaTensor_size(state, batch1, 1), 6,
             "wrong matrix size");
  THArgCheck(THCudaTensor_size(state, t, 2) == THCudaTensor_size(state, batch2, 2), 7,
             "wrong matrix size");
  THArgCheck(THCudaTensor_size(state, batch1, 2) == THCudaTensor_size(state, batch2, 1), 6,
             "wrong matrix size");

  if (t != result) {
    THCudaTensor_resizeAs(state, result, t);
    THCudaTensor_copy(state, result, t);
  }

  bool transpose_result;
  char transpose_batch1, transpose_batch2;
  long lda, ldb, ldc;
  THCudaTensor *result_, *batch1_, *batch2_;
  if (result->stride[1] == 1)
  {
    transpose_result = false;
    result_ = result;
    ldc = result_->stride[2];
  }
  else if (result->stride[2] == 1)
  {
    transpose_result = true;

    THCudaTensor *swap = batch2;
    batch2 = batch1;
    batch1 = swap;

    result_ = result;
    ldc = result_->stride[1];
  }
  else
  {
    transpose_result = false;

    result_ = THCudaTensor_newWithSize3d(state, result->size[0], result->size[2], result->size[1]);
    THCudaTensor_copy(state, result_, result);
    THCudaTensor_transpose(state, result_, NULL, 1, 2);

    ldc = result_->stride[2];
  }

  if (batch1->stride[transpose_result ? 2 : 1] == 1)
  {
    transpose_batch1 = 'n';
    batch1_ = batch1;
    lda = batch1_->stride[transpose_result ? 1 : 2];
  }
  else if (batch1->stride[transpose_result ? 1 : 2] == 1)
  {
    transpose_batch1 = 't';
    batch1_ = batch1;
    lda = batch1_->stride[transpose_result ? 2 : 1];
  }
  else
  {
    transpose_batch1 = transpose_result ? 'n' : 't';
    batch1_ = THCudaTensor_newContiguous(state, batch1);
    lda = batch1_->stride[1];
  }

  if (batch2->stride[transpose_result ? 2 : 1] == 1)
  {
    transpose_batch2 = 'n';
    batch2_ = batch2;
    ldb = batch2_->stride[transpose_result ? 1 : 2];
  }
  else if (batch2->stride[transpose_result ? 1 : 2] == 1)
  {
    transpose_batch2 = 't';
    batch2_ = batch2;
    ldb = batch2_->stride[transpose_result ? 2 : 1];
  }
  else
  {
    transpose_batch2 = transpose_result ? 'n' : 't';
    batch2_ = THCudaTensor_newContiguous(state, batch2);
    ldb = batch2_->stride[1];
  }

  // Compute pointers to matrices in each batch.
  long num_batches = result_->size[0];
  size_t matrices_size = num_batches * sizeof(float*);
  const float **matrices1 = (const float **)THAlloc(matrices_size);
  const float **matrices2 = (const float **)THAlloc(matrices_size);
  float **result_matrices = (float **)THAlloc(matrices_size);
  for (int i = 0; i < num_batches; ++i)
  {
    matrices1[i] = THCudaTensor_data(state, batch1_) + i * batch1_->stride[0];
    matrices2[i] = THCudaTensor_data(state, batch2_) + i * batch2_->stride[0];
    result_matrices[i] = THCudaTensor_data(state, result_) + i * result_->stride[0];
  }

  // Copy pointers to device.
  const float **d_matrices1, **d_matrices2;
  float **d_result_matrices;
  THCudaCheck(cudaMalloc(&d_matrices1, matrices_size));
  THCudaCheck(cudaMalloc(&d_matrices2, matrices_size));
  THCudaCheck(cudaMalloc(&d_result_matrices, matrices_size));

  THCudaCheck(cudaMemcpyAsync(d_matrices1, matrices1, matrices_size, cudaMemcpyHostToDevice));
  THCudaCheck(cudaMemcpyAsync(d_matrices2, matrices2, matrices_size, cudaMemcpyHostToDevice));
  THCudaCheck(cudaMemcpyAsync(d_result_matrices, result_matrices, matrices_size, cudaMemcpyHostToDevice));

  THCudaBlas_gemmBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size[transpose_result ? 2 : 1],
      result_->size[transpose_result ? 1 : 2],
      batch1_->size[transpose_result ? 1 : 2],
      alpha,
      d_matrices1, lda,
      d_matrices2, ldb,
      beta,
      d_result_matrices, ldc,
      num_batches);

  cudaFree(d_matrices1);
  cudaFree(d_matrices2);
  cudaFree(d_result_matrices);
  THFree(matrices1);
  THFree(matrices2);
  THFree(result_matrices);

  if (batch1_ != batch1)
    THCudaTensor_free(state, batch1_);

  if (batch2_ != batch2)
    THCudaTensor_free(state, batch2_);

  if (result_ != result)
    THCudaTensor_freeCopyTo(state, result_, result);
}

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC)                   \
  struct Tensor##NAME##Op {                                             \
    __device__ __forceinline__ void operator()(float* out, float* in) const { \
      *out = CFUNC(*in);                                                \
    }                                                                   \
                                                                        \
    __device__ __forceinline__ void operator()(float* v) const {        \
      *v = CFUNC(*v);                                                   \
    }                                                                   \
  };                                                                    \
                                                                        \
  void THCudaTensor_##NAME(THCState* state, THCudaTensor* self_, THCudaTensor* src) { \
    if (self_ == src) {                                                 \
      if (!THCudaTensor_pointwiseApply1(state, self_, Tensor##NAME##Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING); \
      }                                                                 \
    } else {                                                            \
      THCudaTensor_resizeAs(state, self_, src);                         \
                                                                        \
      if (!THCudaTensor_pointwiseApply2(state, self_, src, Tensor##NAME##Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING); \
      }                                                                 \
    }                                                                   \
                                                                        \
    THCudaCheck(cudaGetLastError());                                    \
  }

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log, log)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log1p, log1p)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(exp, exp)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cos, cos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(acos, acos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cosh, cosh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sin, sin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(asin, asin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sinh, sinh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tan, tan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(atan, atan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tanh, tanh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sqrt, sqrt)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(ceil, ceil)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(floor, floor)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(abs, fabs)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(round, roundf)
