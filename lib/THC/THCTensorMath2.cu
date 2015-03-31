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

struct TensorPowOp {
  TensorPowOp(float v) : val(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = powf(*in, val);
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v = powf(*v, val);
  }

  const float val;
};

void THCudaTensor_pow(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value)
{
  if (self_ == src) {
    if (!THCudaTensor_pointwiseApply1(state, self_, TensorPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THCudaTensor_pointwiseApply2(state, self_, src, TensorPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorTPowOp {
  TensorTPowOp(float v) : val(v) {}

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = powf(val, *in);
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v = powf(val, *v);
  }

  const float val;
};

void THCudaTensor_tpow(THCState *state, THCudaTensor *self_, float value, THCudaTensor *src)
{
  if (self_ == src) {
    if (!THCudaTensor_pointwiseApply1(state, self_, TensorTPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THCudaTensor_pointwiseApply2(state, self_, src, TensorTPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorATan2Op {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = atan2f(*a, *b);
  }
};

void THCudaTensor_atan2(THCState *state, THCudaTensor *self_, THCudaTensor *tx, THCudaTensor *ty)
{
  THArgCheck(THCudaTensor_nElement(state, tx) ==
             THCudaTensor_nElement(state, ty), 3, "sizes do not match");
  THCudaTensor_resizeAs(state, self_, tx);

  if (!THCudaTensor_pointwiseApply3(state, self_, tx, ty, TensorATan2Op())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorClampOp {
  TensorClampOp(float min, float max) : minValue(min), maxValue(max) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = max(min(*in, maxValue), minValue);
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v = max(min(*v, maxValue), minValue);
  }

  const float minValue;
  const float maxValue;
};

void THCudaTensor_clamp(THCState *state, THCudaTensor *self_, THCudaTensor *src, float min_value,
  float max_value)
{
  if (self_ == src) {
    if (!THCudaTensor_pointwiseApply1(state, self_, TensorClampOp(min_value, max_value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THCudaTensor_pointwiseApply2(state, self_, src, TensorClampOp(min_value, max_value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorSignOp {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    float orig = *in;
    *out = (orig > 0) - (orig < 0);
  }

  __device__ __forceinline__ void operator()(float* v) {
    float orig = *v;
    *v = (orig > 0) - (orig < 0);
  }
};

void THCudaTensor_sign(THCState *state, THCudaTensor *self_, THCudaTensor *src)
{
  if (self_ == src) {
    if (!THCudaTensor_pointwiseApply1(state, self_, TensorSignOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCudaTensor_resizeAs(state, self_, src);

    if (!THCudaTensor_pointwiseApply2(state, self_, src, TensorSignOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

float THCudaTensor_meanall(THCState *state, THCudaTensor *self)
{
  THArgCheck(self->nDimension > 0, 1, "empty Tensor");
  return THCudaTensor_sumall(state, self)/THCudaTensor_nElement(state, self);
}

void
THCudaTensor_mean(THCState *state, THCudaTensor *self, THCudaTensor *src, long dim)
{
  THCudaTensor_sum(state, self, src, dim);
  THCudaTensor_div(state, self, self, THCudaTensor_size(state, src, dim));
}

struct square_functor
{
  const float mean;

  square_functor(float mean_) : mean(mean_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return (x-mean)*(x-mean);
  }
};

float THCudaTensor_varall(THCState *state, THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(state, self);
  long size = THCudaTensor_nElement(state, self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));

  float mean = THCudaTensor_meanall(state, self);
  float result = thrust::transform_reduce(self_data, self_data+size, square_functor(mean), (float)0, thrust::plus<float>());

  result = result/(THCudaTensor_nElement(state, self)-1);

  THCudaTensor_free(state, self);
  return result;
}

float THCudaTensor_stdall(THCState *state, THCudaTensor *self)
{
  return sqrt(THCudaTensor_varall(state, self));
}

// Given the sum of values and the sum of squares, compute the variance or standard deviation.
template<bool flag, bool apply_sqrt>
__forceinline__ __device__ float THCudaTensor_computeVar(float sum, float sum2, unsigned row_size) {
  if (flag) {
    sum /= row_size;
    sum2 /= row_size;
    sum2 -= sum * sum;
    sum2 = (sum2 < 0 ? 0 : sum2);
  }
  else {
    sum /= row_size;
    sum2 /= row_size - 1;
    sum2 -= ((float)row_size) / ((float)(row_size - 1)) * sum * sum;
    sum2 = (sum2 < 0 ? 0 : sum2);
  }
  if (apply_sqrt)
    return sqrt(sum2);
  else
    return sum2;
}

/* Compute the variance (or standard deviation) along an outer dimension of a tensor.
 *
 * - num_orows is the size of the flattened outer dimensions;
 * - num_irows is the size of the flattened inner dimensions;
 * - row_size is the size of the dimension along which to compute the variance;
 * - if flag is set, normalize by `row_size` instead of `row_size - 1`
 * - if apply_sqrt is set, compute the standard deviation instead of variance
 *
 * The dimensions to the outside and inside of the specified dimension are considered as flattened.
 * Thread blocks with the same blockIdx.y process an "outer row" (i.e. an element of the flattened
 * outer dimensions, which contains several "inner rows").
 * Each thread processes a single inner row at a time.
 */
template<bool flag, bool apply_sqrt>
__global__ void THCudaTensor_kernel_varOuterDim(float *tgt, float *src_, unsigned num_orows, unsigned num_irows, unsigned row_size)
{
  for (unsigned orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (unsigned irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      float *src = src_ + orow * row_size * num_irows + irow;
      float sum = 0, sum2 = 0;

      for (unsigned col = 0; col < row_size; ++col) {
        float val = *src;
        sum += val;
        sum2 += val * val;

        src += num_irows;
      }

      tgt[orow * num_irows + irow] = THCudaTensor_computeVar<flag, apply_sqrt>(sum, sum2, row_size);
    }
  }
}

template<bool apply_sqrt>
__host__ void THCudaTensor_varOuterDim(THCState *state, THCudaTensor *tgt, THCudaTensor *src, long dimension, int flag)
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

  if (flag) {
    THCudaTensor_kernel_varOuterDim<true, apply_sqrt><<<grid, threads>>>(
        THCudaTensor_data(state, tgt), THCudaTensor_data(state, src), num_orows, num_irows, row_size);
  } else {
    THCudaTensor_kernel_varOuterDim<false, apply_sqrt><<<grid, threads>>>(
        THCudaTensor_data(state, tgt), THCudaTensor_data(state, src), num_orows, num_irows, row_size);
  }
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}


/* Compute the variance (or standard deviation) of the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 * - if flag is set, normalize by `row_size` instead of `row_size - 1`
 * - if apply_sqrt is set, compute the standard deviation instead of variance
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<bool flag, bool apply_sqrt>
__global__ void THCudaTensor_kernel_varInnermostDim(float *tgt, float *src_, unsigned num_rows, unsigned row_size)
{
  __shared__ float ssum[32][16];
  __shared__ float ssum2[32][16];

  for (unsigned block_row = blockIdx.x * blockDim.y; block_row < num_rows; block_row += blockDim.y * gridDim.x) {
    unsigned row = block_row + threadIdx.y;
    float sum = 0, sum2 = 0;
    if (row < num_rows) {
      float *src = src_ + row * row_size;
      // Sequential reduction within a thread.
      for (unsigned col = threadIdx.x; col < row_size; col += blockDim.x) {
        float val = src[col];
        sum += val;
        sum2 += val * val;
      }
    }
    ssum[threadIdx.y][threadIdx.x] = sum;
    ssum2[threadIdx.y][threadIdx.x] = sum2;
    __syncthreads();

    // Reduce intermediate values to single value.
    for (unsigned s = 8; s > 1; s >>= 1) {
      if (row < num_rows && threadIdx.x < s) {
        ssum[threadIdx.y][threadIdx.x] += ssum[threadIdx.y][threadIdx.x + s];
        ssum2[threadIdx.y][threadIdx.x] += ssum2[threadIdx.y][threadIdx.x + s];
      }
      __syncthreads();
    }

    if (row < num_rows && threadIdx.x == 0) {
      sum = ssum[threadIdx.y][0] + ssum[threadIdx.y][1];
      sum2 = ssum2[threadIdx.y][0] + ssum2[threadIdx.y][1];
      tgt[row] = THCudaTensor_computeVar<flag, apply_sqrt>(sum, sum2, row_size);
    }
    __syncthreads();
  }
}

template<bool apply_sqrt>
__host__ void THCudaTensor_varInnermostDim(THCState *state, THCudaTensor *tgt, THCudaTensor *src, int flag)
{
  unsigned ndim = THCudaTensor_nDimension(state, src);
  // Treat all outer dimensions as a single dimension.
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THCudaTensor_size(state, src, dim);
  }
  unsigned row_size = THCudaTensor_size(state, src, ndim - 1);

  // From limited testing, 16x32 seemed a good compromise for handling both long and short dimensions.
  dim3 threads(16, 32);
  dim3 grid(min(1024, DIVUP(num_rows, threads.y)));

  if (flag) {
    THCudaTensor_kernel_varInnermostDim<true, apply_sqrt><<<grid, threads>>>(
        THCudaTensor_data(state, tgt), THCudaTensor_data(state, src), num_rows, row_size);
  } else {
    THCudaTensor_kernel_varInnermostDim<false, apply_sqrt><<<grid, threads>>>(
        THCudaTensor_data(state, tgt), THCudaTensor_data(state, src), num_rows, row_size);
  }
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

void THCudaTensor_var(THCState *state, THCudaTensor *self_, THCudaTensor *src, long dimension, int flag)
{
  THLongStorage *dim = THCudaTensor_newSizeOf(state, src);
  THLongStorage_set(dim, dimension, 1);
  THCudaTensor_resize(state, self_, dim, NULL);
  THLongStorage_free(dim);

  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  src = THCudaTensor_newContiguous(state, src);

  if (dimension == THCudaTensor_nDimension(state, src) - 1) {
    THCudaTensor_varInnermostDim<false>(state, self, src, flag);
  } else {
    THCudaTensor_varOuterDim<false>(state, self, src, dimension, flag);
  }

  THCudaTensor_free(state, src);
  THCudaTensor_freeCopyTo(state, self, self_);
}

void THCudaTensor_std(THCState *state, THCudaTensor *self_, THCudaTensor *src, long dimension, int flag)
{
  THLongStorage *dim = THCudaTensor_newSizeOf(state, src);
  THLongStorage_set(dim, dimension, 1);
  THCudaTensor_resize(state, self_, dim, NULL);
  THLongStorage_free(dim);

  THCudaTensor *self = THCudaTensor_newContiguous(state, self_);
  src = THCudaTensor_newContiguous(state, src);

  if (dimension == THCudaTensor_nDimension(state, src) - 1) {
    THCudaTensor_varInnermostDim<true>(state, self, src, flag);
  } else {
    THCudaTensor_varOuterDim<true>(state, self, src, dimension, flag);
  }

  THCudaTensor_free(state, src);
  THCudaTensor_freeCopyTo(state, self, self_);
}


template<class Op>
void THCudaTensor_logicalValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, Op op)
{
  THCudaTensor_resizeAs(state, self_, src);

  if (!THCudaTensor_pointwiseApply2(state, self_, src, op)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorLTValueOp {
  TensorLTValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in < value);
  }

  const float value;
};

void THCudaTensor_ltValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(state, self_, src, TensorLTValueOp(value));
}

struct TensorGTValueOp {
  TensorGTValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in > value);
  }

  const float value;
};

void THCudaTensor_gtValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(state, self_, src, TensorGTValueOp(value));
}

struct TensorLEValueOp {
  TensorLEValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in <= value);
  }

  const float value;
};

void THCudaTensor_leValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(state, self_, src, TensorLEValueOp(value));
}

struct TensorGEValueOp {
  TensorGEValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in >= value);
  }

  const float value;
};

void THCudaTensor_geValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(state, self_, src, TensorGEValueOp(value));
}

struct TensorEQValueOp {
  TensorEQValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in == value);
  }

  const float value;
};

void THCudaTensor_eqValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(state, self_, src, TensorEQValueOp(value));
}

struct TensorNEValueOp {
  TensorNEValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in != value);
  }

  const float value;
};

void THCudaTensor_neValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value)
{
  THCudaTensor_logicalValue(state, self_, src, TensorNEValueOp(value));
}

template<class Op>
void THCudaTensor_logicalTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2, Op op)
{
  THCudaTensor_resizeAs(state, self_, src1);
  THArgCheck(THCudaTensor_nElement(state, src1) == THCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (!THCudaTensor_pointwiseApply3(state, self_, src1, src2, op)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorLTOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a < *b);
  }
};

struct TensorGTOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a > *b);
  }
};

struct TensorLEOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a <= *b);
  }
};

struct TensorGEOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a >= *b);
  }
};

struct TensorEQOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a == *b);
  }
};

struct TensorNEOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a != *b);
  }
};

void THCudaTensor_ltTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(state, self_, src1, src2, TensorLTOp());
}


void THCudaTensor_gtTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(state, self_, src1, src2, TensorGTOp());
}


void THCudaTensor_leTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(state, self_, src1, src2, TensorLEOp());
}


void THCudaTensor_geTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(state, self_, src1, src2, TensorGEOp());
}


void THCudaTensor_eqTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(state, self_, src1, src2, TensorEQOp());
}


void THCudaTensor_neTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2)
{
  THCudaTensor_logicalTensor(state, self_, src1, src2, TensorNEOp());
}


struct norm_functor
{
  const float exponent;

  norm_functor(float exponent_) : exponent(exponent_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return pow(fabs(x), exponent);
  }
};

struct partial_not_equal_functor
{
  const float rhs;
  partial_not_equal_functor(float rhs) : rhs(rhs) {}
  __host__ __device__ bool operator()(const float &lhs) const {return lhs != rhs;}
};

float THCudaTensor_normall(THCState *state, THCudaTensor *self, float value)
{
  self = THCudaTensor_newContiguous(state, self);
  long size = THCudaTensor_nElement(state, self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));

  float result;
  if(value == 0.0f) {
    result = thrust::transform_reduce(self_data, self_data+size, partial_not_equal_functor(0.0f), (float)0, thrust::plus<float>());
  } else {
    result = thrust::transform_reduce(self_data, self_data+size, norm_functor(value), (float)0, thrust::plus<float>());
    result = pow(result, (float)1.0/value);
  }

  THCudaTensor_free(state, self);
  return result;
}

void THCudaTensor_norm(THCState *state, THCudaTensor* self, THCudaTensor* src, float value, long dimension)
{
  if (value == 0.0f) {
    THCudaTensor_reduceDim(state, self, src,
                           partial_not_equal_functor(0.0f), thrust::plus<float>(),
                           0.0f, dimension);
  } else {
    THCudaTensor_reduceDim(state, self, src,
                           norm_functor(value), thrust::plus<float>(),
                           0.0f, dimension);
    THCudaTensor_pow(state, self, self, 1/value);
  }

  THCudaCheck(cudaGetLastError());
}

__global__ void THCudaTensor_kernel_renorm(float *data, const float value, const long size, const float maxnorm)
{
  __shared__ float buffer[32];
  long tx = threadIdx.x;
  long bx = blockIdx.x;
  long step = blockDim.x;
  float *row = data + size*bx;

  buffer[tx] = 0;

  // get norm of axis
  for (long i=tx; i<size; i+=step)
  {
    buffer[tx] += pow(fabs(row[i]), value);
  }
  // add (reduce)
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if (tx < stride)
      buffer[tx] += buffer[tx+stride];
  }
  // clip norms
  __syncthreads();
  float norm = pow(buffer[0], 1/value);
  if (norm > maxnorm)
  {
    norm = maxnorm / (norm + 1e-7);
    // renormalize
    for (long i=tx; i<size; i+=step)
    {
      row[i] *= norm;
    }
  }
}

void THCudaTensor_renorm(THCState *state, THCudaTensor* self, THCudaTensor* src, float value, long dimension, float maxnorm)
{
  THCudaTensor *self_;
  THCudaTensor *src_ = THCudaTensor_newTranspose(state, src, dimension, 0);
  THCudaTensor *data = THCudaTensor_newClone(state, src_);
  long size = THCudaTensor_nElement(state, data)/data->size[0];

  THArgCheck(dimension >= 0 && dimension < THCudaTensor_nDimension(state, src), 3, "invalid dimension");
  THArgCheck(value > 0, 2, "non-positive-norm not supported");
  THArgCheck(THCudaTensor_nDimension(state, src) > 1, 1, "need at least 2 dimensions");

  dim3 grid(data->size[0]);
  dim3 threads(32);

  THCudaTensor_kernel_renorm<<<grid, threads>>>(THCudaTensor_data(state, data), value, size, maxnorm);

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, src_);
  self_ = THCudaTensor_newTranspose(state, data, dimension, 0);
  THCudaTensor_resizeAs(state, self, self_);
  THCudaTensor_freeCopyTo(state, self_, self);
  THCudaTensor_free(state, data);
}

struct dist_functor
{
  const float exponent;

  dist_functor(float exponent_) : exponent(exponent_) {}

  __host__ __device__ float operator()(const float& x, const float& y) const
  {
    return pow(fabs(x-y), exponent);
  }
};

float THCudaTensor_dist(THCState *state, THCudaTensor *self, THCudaTensor *src, float value)
{
  self = THCudaTensor_newContiguous(state, self);
  long size = THCudaTensor_nElement(state, self);
  src = THCudaTensor_newContiguous(state, src);
  thrust::device_ptr<float> self_data(THCudaTensor_data(state, self));
  thrust::device_ptr<float> src_data(THCudaTensor_data(state, src));

  float result = thrust::inner_product(self_data, self_data+size, src_data, (float) 0,thrust::plus<float>(), dist_functor(value));

  THCudaTensor_free(state, src);
  THCudaTensor_free(state, self);

  return pow(result, (float)1.0/value);
}

void THCudaTensor_rand(THCState *state, THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_uniform(state, r_, 0, 1);
}

void THCudaTensor_randn(THCState *state, THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(state, r_, size, NULL);
  THCudaTensor_normal(state, r_, 0, 1);
}

__global__ void THCudaTensor_kernel_indexFill(
   float *tensor, long* stride, float *index, long src_nDim,
   int dim, long idx_size, long tensor_size, long size_dim, float val
)
{
  int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  long flat_size = tensor_size / idx_size;

  if (thread_idx < flat_size)
  {
    long coeff = 0;
    for (int i=0; i<idx_size; i++)
    {
      int leftover = thread_idx;
      int srcIdx = 0;
      for (int d=0; d<src_nDim; d++)
      {
        if (d < dim)
        {
          coeff = leftover / (stride[d] / size_dim);
          leftover -= coeff * (stride[d] / size_dim);
          srcIdx += coeff * stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / stride[d];
          leftover -= coeff * stride[d];
          srcIdx += coeff * stride[d];
        }
      }
        tensor[srcIdx + (int)((index[i])-1)*stride[dim]] = val;
    }
  }
}

__global__ void THCudaTensor_kernel_indexCopy(
   float *res, float *src, long* res_stride, float *index,
   long res_nDim, int dim, long idx_size, long src_size, long size_dim
)
{
  int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  long flat_size = src_size / idx_size;

  if (thread_idx < flat_size)
  {
    long coeff = 0;
    for (int i=0; i<idx_size; i++)
    {
      int leftover = thread_idx;
      int targetIdx = 0;
      int resIdx = 0;
      for (int d=0; d<res_nDim; d++)
      {
        if (d < dim)
        {
          long stride_d = res_stride[d] / size_dim;
          coeff = leftover / stride_d;
          leftover -= coeff * stride_d;
          targetIdx += coeff * stride_d * idx_size;
          resIdx += coeff * res_stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / res_stride[d];
          leftover -= coeff * res_stride[d];
          targetIdx += coeff * res_stride[d];
          resIdx += coeff * res_stride[d];
        }
      }
      res[resIdx + ((int)(index[i])-1)*res_stride[dim]] = src[targetIdx + i*res_stride[dim]];
    }
  }
}

void THCudaTensor_indexCopy(THCState *state, THCudaTensor *res_, int dim, THLongTensor *indices, THCudaTensor *src)
{
  THCudaTensor *indices_;
  long *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");
  THArgCheck(nIndex == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  src = THCudaTensor_newContiguous(state, src);
  indices_ = THCudaTensor_newWithSize1d(state, nIndex);
  THCudaTensor_copyLong(state, indices_, indices);

  nRes = THCudaTensor_nElement(state, res_);
  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  THCudaCheck(cudaMalloc((void**)&stride_, res_->nDimension * sizeof(long)));
  THCudaCheck(cudaMemcpy(stride_, res_->stride, res_->nDimension * sizeof(long), cudaMemcpyHostToDevice));

  THCudaTensor_kernel_indexCopy<<<nblocks, nthreads>>>(
    THCudaTensor_data(state, res_), THCudaTensor_data(state, src),
    stride_, THCudaTensor_data(state, indices_),
    res_->nDimension, dim, nIndex,
    THCudaTensor_nElement(state, src), res_->size[dim]
  );

  THCudaCheck(cudaFree(stride_));
  THCudaTensor_free(state, indices_);
  THCudaTensor_free(state, src);
}


void THCudaTensor_indexFill(THCState *state, THCudaTensor *res_, int dim, THLongTensor *indices, float val)
{
  THCudaTensor *indices_;
  long *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < res_->nDimension,4,"Indexing dim is out of bounds");
  THArgCheck(res_->nDimension > 0, 2, "Source tensor is empty");

  indices_ = THCudaTensor_newWithSize1d(state, nIndex);
  THCudaTensor_copyLong(state, indices_, indices);

  nRes = THCudaTensor_nElement(state, res_) / res_->size[dim] * nIndex;


  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  THCudaCheck(cudaMalloc((void**)&stride_, res_->nDimension * sizeof(long)));
  THCudaCheck(cudaMemcpy(stride_, res_->stride, res_->nDimension * sizeof(long), cudaMemcpyHostToDevice));

  THCudaTensor_kernel_indexFill<<<nblocks, nthreads>>>(
    THCudaTensor_data(state, res_), stride_, THCudaTensor_data(state, indices_),
    res_->nDimension, dim, nIndex, nRes, res_->size[dim], val
  );

  THCudaCheck(cudaFree(stride_));
  THCudaTensor_free(state, indices_);
}

__global__ void THCudaTensor_kernel_indexSelect(
   float *tensor, float *src, long* src_stride, float *index,
   long src_nDim, int dim, long idx_size, long tensor_size, long size_dim
)
{
  int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  long flat_size = tensor_size / idx_size;

  if (thread_idx < flat_size)
  {
    long coeff = 0;
    for (int i=0; i<idx_size; i++)
    {
      int leftover = thread_idx;
      int targetIdx = 0;
      int srcIdx = 0;
      for (int d=0; d<src_nDim; d++)
      {
        if (d < dim)
        {
          long stride_d = src_stride[d] / size_dim;
          coeff = leftover / stride_d;
          leftover -= coeff * stride_d;
          targetIdx += coeff * stride_d * idx_size;
          srcIdx += coeff * src_stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / src_stride[d];
          leftover -= coeff * src_stride[d];
          targetIdx += coeff * src_stride[d];
          srcIdx += coeff * src_stride[d];
        }
      }
      tensor[targetIdx + i*src_stride[dim]] = src[srcIdx + ((int)(index[i])-1)*src_stride[dim]];
    }
  }
}


void THCudaTensor_indexSelect(THCState *state, THCudaTensor *res_, THCudaTensor *src, int dim, THLongTensor *indices)
{
  THLongStorage *newSize;
  THCudaTensor *indices_;
  long *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");

  newSize = THLongStorage_newWithSize(src->nDimension);
  THLongStorage_rawCopy(newSize, src->size);
  newSize->data[dim] = nIndex;
  THCudaTensor_resize(state, res_, newSize, NULL);
  THLongStorage_free(newSize);

  indices_ = THCudaTensor_newWithSize1d(state, nIndex);
  THCudaTensor_copyLong(state, indices_, indices);

  nRes = THCudaTensor_nElement(state, res_);
  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  THCudaCheck(cudaMalloc((void**)&stride_, src->nDimension * sizeof(long)));
  THCudaCheck(cudaMemcpy(stride_, src->stride, src->nDimension * sizeof(long), cudaMemcpyHostToDevice));

  THCudaTensor_kernel_indexSelect<<<nblocks, nthreads>>>(
    THCudaTensor_data(state, res_), THCudaTensor_data(state, src),
    stride_, THCudaTensor_data(state, indices_),
    src->nDimension, dim, indices->size[0], nRes, src->size[dim]
  );

  THCudaCheck(cudaFree(stride_));
  THCudaTensor_free(state, indices_);
}

struct TensorMaskedFillOp {
  TensorMaskedFillOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* t, float* mask) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    const float maskVal = *mask;
    if (maskVal != 0.0f) {
      *t = value;
    }
  }

  float value;
};

void THCudaTensor_maskedFill(THCState* state,
                             THCudaTensor *tensor, THCudaTensor *mask, float value)
{
  THArgCheck(THCudaTensor_nElement(state, tensor) ==
             THCudaTensor_nElement(state, mask),
             2, "sizes do not match");

  if (!THCudaTensor_pointwiseApply2(state, tensor, mask, TensorMaskedFillOp(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorMaskedCopyOp {
  __device__ __forceinline__ void operator()(float* out, float* mask, float* in) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      *out = *in;
    }
  }
};

void THCudaTensor_maskedCopy(THCState* state,
                             THCudaTensor *tensor, THCudaTensor *mask, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(state, mask) ==
             THCudaTensor_nElement(state, src),
             2, "sizes do not match");

  THCudaTensor_resizeAs(state, tensor, src);

  if (!THCudaTensor_pointwiseApply3(state, tensor, mask, src, TensorMaskedCopyOp())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorMaskedSelectOp {
  TensorMaskedSelectOp(float* t) : out(t) {}
  __device__ __forceinline__ void operator()(float* mask, float* maskPrefixSum, float* in) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      out[(int) *maskPrefixSum] = *in;
    }
  }

  float* out;
};

void THCudaTensor_maskedSelect(THCState* state,
                               THCudaTensor *tensor, THCudaTensor *src, THCudaTensor *mask)
{
  THArgCheck(THCudaTensor_nElement(state, mask) == THCudaTensor_nElement(state, src),
             2, "sizes do not match");

  // Determine our output size
  THCudaTensor* contigMask = THCudaTensor_newContiguous(state, mask);
  int totalElements = (int) THCudaTensor_sumall(state, contigMask);
  THCudaTensor_resize1d(state, tensor, totalElements);

  // Use a prefix sum to determine the output locations of the masked elements
  THCudaTensor* maskPrefixSum = THCudaTensor_new(state);
  THCudaTensor_resizeAs(state, maskPrefixSum, mask);

  thrust::device_ptr<float>
    maskData(THCudaTensor_data(state, contigMask));
  thrust::device_ptr<float>
    maskPrefixSumData(THCudaTensor_data(state, maskPrefixSum));
  thrust::exclusive_scan(maskData,
                         maskData + THCudaTensor_nElement(state, contigMask),
                         maskPrefixSumData);

  // Then copy over the masked elements at their desired output index
  bool status = THCudaTensor_pointwiseApply3(
    state, contigMask, maskPrefixSum,
    src, TensorMaskedSelectOp(THCudaTensor_data(state, tensor)));

  THCudaTensor_free(state, contigMask);
  THCudaTensor_free(state, maskPrefixSum);

  if (!status) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_maskedFillByte(THCState* state, THCudaTensor *tensor, THByteTensor *mask, float value)
{
  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
  THCudaTensor* maskCuda = THCudaTensor_newWithSize(state, maskSize, NULL);
  THLongStorage_free(maskSize);
  THCudaTensor_copyByte(state, maskCuda, mask);
  THCudaTensor_maskedFill(state, tensor, maskCuda, value);
  THCudaTensor_free(state, maskCuda);
}

void THCudaTensor_maskedCopyByte(THCState* state, THCudaTensor *tensor, THByteTensor *mask, THCudaTensor *src)
{
  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
  THCudaTensor* maskCuda = THCudaTensor_newWithSize(state, maskSize, NULL);
  THLongStorage_free(maskSize);
  THCudaTensor_copyByte(state, maskCuda, mask);
  THCudaTensor_maskedCopy(state, tensor, maskCuda, src);
  THCudaTensor_free(state, maskCuda);
}

void THCudaTensor_maskedSelectByte(THCState* state, THCudaTensor *tensor, THCudaTensor *src, THByteTensor *mask)
{
  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
  THCudaTensor* maskCuda = THCudaTensor_newWithSize(state, maskSize, NULL);
  THLongStorage_free(maskSize);
  THCudaTensor_copyByte(state, maskCuda, mask);
  THCudaTensor_maskedSelect(state, tensor, src, maskCuda);
  THCudaTensor_free(state, maskCuda);
}
