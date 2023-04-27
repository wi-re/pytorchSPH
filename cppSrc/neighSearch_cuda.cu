
#ifdef __INTELLISENSE__
#define __CUDACC__
#define __device__
#endif


#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <utility>
#include <vector>

namespace {
// template <typename scalar_t>
// __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
//   return 1.0 / (1.0 + exp(-z));
// }

// template <typename scalar_t>
// __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
//   const auto s = sigmoid(z);
//   return (1.0 - s) * s;
// }

// template <typename scalar_t>
// __device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
//   const auto t = tanh(z);
//   return 1 - (t * t);
// }

// template <typename scalar_t>
// __device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
//   return fmaxf(0.0, z) + fminf(0.0, alpha * (exp(z) - 1.0));
// }

// template <typename scalar_t>
// __device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
//   const auto e = exp(z);
//   const auto d_relu = z < 0.0 ? 0.0 : 1.0;
//   return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
// }

// template <typename scalar_t>
// __global__ void lltm_cuda_forward_kernel(
//     const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
//     const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
//     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
//     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
//     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
//     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
//     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell) {
//   //batch index
//   const int n = blockIdx.y;
//   // column index
//   const int c = blockIdx.x * blockDim.x + threadIdx.x;
//   if (c < gates.size(2)){
//     input_gate[n][c] = sigmoid(gates[n][0][c]);
//     output_gate[n][c] = sigmoid(gates[n][1][c]);
//     candidate_cell[n][c] = elu(gates[n][2][c]);
//     new_cell[n][c] =
//         old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
//     new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
//   }
// }

using int2Ptr_t = torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits>;
using intPtr_t = torch::PackedTensorAccessor32<int32_t,1,torch::RestrictPtrTraits>;
template<typename scalar_t>
using scalar2Ptr_t = torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits>;
template<typename scalar_t>
using scalarPtr_t = torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits>;

__device__ ::std::pair<int64_t, int64_t> queryHashMapCUDA(int32_t qIDx, int32_t qIDy, 
  const int2Ptr_t hashTable, const intPtr_t cellIndices, const intPtr_t cumCell,  const intPtr_t cellSpan, int32_t cellsX, 
  int32_t hashMapLength){
    
    if(qIDx < 0 || qIDy < 0) return {-1,-1};
    auto qLin = qIDx + cellsX * qIDy;
    auto qHash = (qIDx * 3 +  qIDy * 5)%  hashMapLength;
    auto hashEntries = hashTable[qHash];
    if(hashEntries[0] == -1)
      return {-1,-1};
    auto hashIndices = hashEntries[0];
    auto minIter = hashEntries[0];
    auto maxIter = (hashEntries[0] + hashEntries[1]);
    for(int32_t i = minIter; i < maxIter; i++){
        auto hashIndex = i;
        auto cellIndex = cellIndices[hashIndex];
        if(cellIndex == qLin){
            auto minCellIter = cellSpan[hashIndex];
            auto maxCellIter = (cellSpan[hashIndex] + cumCell[hashIndex]);  
            return {minCellIter,maxCellIter};
        }
    }
    return {-1,-1}; 
}

template <typename scalar_t>
__global__ void neighborSearchCUDAKernel(
    const scalar2Ptr_t<scalar_t> queryParticles,
    const scalarPtr_t<scalar_t> support,
    const scalar2Ptr_t<scalar_t> sortedParticles,
    const scalarPtr_t<scalar_t> sortedSupport,
    const int2Ptr_t hashTable,
    const intPtr_t cellIndices,
    const intPtr_t cumCell,
    const intPtr_t cellSpan,
    const intPtr_t sort,
    intPtr_t counter,
    const scalarPtr_t<scalar_t> qMin,
    float hMax,
    const int32_t cellsX, int32_t hashMapLength, int32_t numParticles, int32_t searchRadius) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= numParticles)
    return;
    
      auto xi = queryParticles[idx];
      auto hi = support[idx];
      int32_t qIDx = ceil((xi[0] - qMin[0]) / hMax);
      int32_t qIDy = ceil((xi[1] - qMin[1]) / hMax);

      int32_t numNeighbors = 0;

      for (int32_t xx = -searchRadius; xx<= searchRadius; ++xx){
        for (int32_t yy = -searchRadius; yy<= searchRadius; ++yy){
          auto currentIndexPair = queryHashMapCUDA(qIDx + xx, qIDy + yy, hashTable, cellIndices, cumCell, cellSpan, cellsX, hashMapLength);
          if(currentIndexPair.first == -1) continue;
          for(int32_t j = currentIndexPair.first; j < currentIndexPair.second; ++j){
            auto xj = sortedParticles[j];
            auto dist = sqrt((xi[0] - xj[0]) * (xi[0] - xj[0]) + (xi[1] - xj[1]) * (xi[1] - xj[1]));
            if( dist < hi){
              ++numNeighbors;
              // rows.push_back(b);
              // cols.push_back(sort[j]);
            }
          }
        }
      }

    counter[idx] = numNeighbors;
    }

        // queryParticles_.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        // support_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        // sortedParticles.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        // sortedSupport.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        // hashTable_.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
        // cellIndices_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
        // cumCell_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
        // cellSpan_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
        // sort_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
        // qMin_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),    
        // offset.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),      
        // neighborListI.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),      
        // neighborListJ.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),        
template <typename scalar_t>
__global__ void constructNeighborhoodsCUDA(
    const scalar2Ptr_t<scalar_t> queryParticles,
    const scalarPtr_t<scalar_t> support,
    const scalar2Ptr_t<scalar_t> sortedParticles,
    const scalarPtr_t<scalar_t> sortedSupport,
    const int2Ptr_t hashTable,
    const intPtr_t cellIndices,
    const intPtr_t cumCell,
    const intPtr_t cellSpan,
    const intPtr_t sort,
    const scalarPtr_t<scalar_t> qMin,
    const intPtr_t offsets,
    intPtr_t neighborListI,
    intPtr_t neighborListJ,
    float hMax,
    const int32_t cellsX, int32_t hashMapLength, int32_t numParticles, int32_t searchRadius) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= numParticles)
    return;
    
      auto xi = queryParticles[idx];
      auto hi = support[idx];
      auto offset = idx == 0 ? 0 : offsets[idx-1];
      int32_t qIDx = ceil((xi[0] - qMin[0]) / hMax);
      int32_t qIDy = ceil((xi[1] - qMin[1]) / hMax);

      int32_t numNeighbors = 0;

      for (int32_t xx = -searchRadius; xx<= searchRadius; ++xx){
        for (int32_t yy = -searchRadius; yy<= searchRadius; ++yy){
          auto currentIndexPair = queryHashMapCUDA(qIDx + xx, qIDy + yy, hashTable, cellIndices, cumCell, cellSpan, cellsX, hashMapLength);
          if(currentIndexPair.first == -1) continue;
          for(int32_t j = currentIndexPair.first; j < currentIndexPair.second; ++j){
            auto xj = sortedParticles[j];
            auto dist = sqrt((xi[0] - xj[0]) * (xi[0] - xj[0]) + (xi[1] - xj[1]) * (xi[1] - xj[1]));
            if( dist < hi){
              neighborListI[offset + numNeighbors] = idx;
              neighborListJ[offset + numNeighbors] = sort[j];
              // rows.push_back(b);
              // cols.push_back(sort[j]);
              ++numNeighbors;
            }
          }
        }
      }

    // counter[idx] = numNeighbors;
    }
  
} // namespace


std::pair<torch::Tensor,torch::Tensor> countNeighborsCUDAImpl(
    torch::Tensor queryParticles_, torch::Tensor support_,
    torch::Tensor sortedParticles, torch::Tensor sortedSupport,
    torch::Tensor hashTable_,
    torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
    torch::Tensor sort_,
    torch::Tensor qMin_,
    float hMax,
    int32_t cellsX, int32_t hashMapLength, int32_t searchRadius){
      auto counter = torch::zeros({support_.size(0)}, torch::TensorOptions()
          .dtype(torch::kInt32)
          .layout(torch::kStrided)
        .device(torch::kCUDA, queryParticles_.get_device()));

        auto numParticles = queryParticles_.size(0);
        // std::cout << "Number of particles: " << numParticles << std::endl;

  int32_t threads = 1024;
  int32_t blocks = (int32_t) floor(numParticles / threads) + (numParticles % threads == 0 ? 0 : 1);
  // std::cout << "Launching with: " << blocks << " @ " << threads << " tpb for " << numParticles << " particles." << std::endl;

  AT_DISPATCH_FLOATING_TYPES(queryParticles_.type(), "lltm_forward_cuda", ([&] {
    neighborSearchCUDAKernel<scalar_t><<<blocks, threads>>>(
        queryParticles_.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        support_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        sortedParticles.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        sortedSupport.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        hashTable_.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
        cellIndices_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
        cumCell_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
        cellSpan_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
        sort_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
        counter.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
        qMin_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),        
        hMax, cellsX, hashMapLength, numParticles, searchRadius);
  }));
  auto offsets  = torch::cumsum(counter, 0);
  offsets = offsets.to(torch::kInt);
        return {counter, offsets};
    }

std::pair<torch::Tensor,torch::Tensor> constructNeighborsCUDAImpl(
    torch::Tensor queryParticles_, torch::Tensor support_,
    torch::Tensor sortedParticles, torch::Tensor sortedSupport,
    torch::Tensor hashTable_,
    torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
    torch::Tensor sort_,
    torch::Tensor qMin_,
    torch::Tensor counter,
    torch::Tensor offset,
    float hMax,
    int32_t cellsX, int32_t hashMapLength, int32_t searchRadius){

      int32_t numElements = torch::max(offset).item<int32_t>();

      auto neighborListI = torch::zeros({numElements}, torch::TensorOptions()
          .dtype(torch::kInt32)
          .layout(torch::kStrided)
        .device(torch::kCUDA, queryParticles_.get_device()));
      auto neighborListJ = torch::zeros({numElements}, torch::TensorOptions()
          .dtype(torch::kInt32)
          .layout(torch::kStrided)
        .device(torch::kCUDA, queryParticles_.get_device()));



        auto numParticles = queryParticles_.size(0);
        // std::cout << "Number of particles: " << numParticles << std::endl;

  int32_t threads = 1024;
  int32_t blocks = (int32_t) floor(numParticles / threads) + (numParticles % threads == 0 ? 0 : 1);
  // std::cout << "Launching with: " << blocks << " @ " << threads << " tpb for " << numParticles << " particles." << std::endl;
  

  AT_DISPATCH_FLOATING_TYPES(queryParticles_.type(), "lltm_forward_cuda", ([&] {
    constructNeighborhoodsCUDA<scalar_t><<<blocks, threads>>>(
        queryParticles_.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        support_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        sortedParticles.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        sortedSupport.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        hashTable_.packed_accessor32<int32_t,2,torch::RestrictPtrTraits>(),
        cellIndices_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
        cumCell_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
        cellSpan_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
        sort_.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),
        qMin_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),    
        offset.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),      
        neighborListI.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),      
        neighborListJ.packed_accessor32<int32_t,1,torch::RestrictPtrTraits>(),        
        hMax, cellsX, hashMapLength, numParticles, searchRadius);
  }));
  // auto offsets  = torch::cumsum(counter, 0) - counter[0];
        return {neighborListI, neighborListJ};
    }

// std::vector<torch::Tensor> lltm_cuda_forward(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor bias,
//     torch::Tensor old_h,
//     torch::Tensor old_cell) {
//   auto X = torch::cat({old_h, input}, /*dim=*/1);
//   auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

//   const auto batch_size = old_cell.size(0);
//   const auto state_size = old_cell.size(1);

//   auto gates = gate_weights.reshape({batch_size, 3, state_size});
//   auto new_h = torch::zeros_like(old_cell);
//   auto new_cell = torch::zeros_like(old_cell);
//   auto input_gate = torch::zeros_like(old_cell);
//   auto output_gate = torch::zeros_like(old_cell);
//   auto candidate_cell = torch::zeros_like(old_cell);

//   const int threads = 1024;
//   const dim3 blocks((state_size + threads - 1) / threads, batch_size);

//   AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
//     lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
//         gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
//         old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
//         new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
//         new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
//         input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
//         output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
//         candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
//   }));

//   return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
// }
 
// std::vector<torch::Tensor> lltm_cuda_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell,
//     torch::Tensor new_cell,
//     torch::Tensor input_gate,
//     torch::Tensor output_gate,
//     torch::Tensor candidate_cell,
//     torch::Tensor X,
//     torch::Tensor gates,
//     torch::Tensor weights) {
//   auto d_old_cell = torch::zeros_like(new_cell);
//   auto d_gates = torch::zeros_like(gates);

//   const auto batch_size = new_cell.size(0);
//   const auto state_size = new_cell.size(1);

//   const int threads = 1024;
//   const dim3 blocks((state_size + threads - 1) / threads, batch_size);

//   AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_forward_cuda", ([&] {
//     lltm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
//         d_old_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
//         d_gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
//         grad_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
//         grad_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
//         new_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
//         input_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
//         output_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
//         candidate_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
//         gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
//   }));

//   auto d_gate_weights = d_gates.flatten(1, 2);
//   auto d_weights = d_gate_weights.t().mm(X);
//   auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

//   auto d_X = d_gate_weights.mm(weights);
//   auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
//   auto d_input = d_X.slice(/*dim=*/1, state_size);

//   return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
// }