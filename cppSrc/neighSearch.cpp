// #define _OPENMP
#include <algorithm>
#include <ATen/Parallel.h>
#include <ATen/ParallelOpenMP.h>
// #include <ATen/ParallelNativeTBB.h>
#include <torch/extension.h>

#include <vector>

/*
std::vector<torch::Tensor> sortPointSet( torch::Tensor points, torch::Tensor supports){
  auto hMax = at::max(supports);
  // std::cout << "Output from pytorch module" << std::endl;
  // std::cout << "hMax " << hMax << std::endl;
  auto qMin = std::get<0>(at::min(points,0)) - hMax;
  auto qMax = std::get<0>(at::max(points,0)) + 2 * hMax;
  // std::cout << "qMin " << qMin << std::endl;
  // std::cout << "qMax " << qMax << std::endl;

  auto qEx = qMax - qMin;
  // std::cout << "qEx: " << qEx;
  
  auto cells = at::ceil(qEx / hMax).to(torch::kInt);
  // std::cout << "Cells: " << cells;
  auto indices = at::ceil((points - qMin) / hMax).to(torch::kInt);

  // auto linearIndices = at::empty({points.size(0)}, torch::TensorOptions().dtype(torch::kInt));

  auto linearIndices = indices.index({torch::indexing::Slice(), 0}) + cells[0] * indices.index({torch::indexing::Slice(), 1});
  // std::cout << __FILE__ << " " << __LINE__ << ": " << linearIndices << std::endl;

  auto indexAccessor = indices.accessor<int32_t, 2>();
  auto linearIndexAccessor = linearIndices.accessor<int32_t, 1>();
  auto cols = cells[0].item<int32_t>();
  int32_t batch_size = indices.size(0); 
  // at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end) {
  //   for (int32_t b = start; b < end; b++) {
  //     linearIndexAccessor[b] = indexAccessor[b][0] + cols * indexAccessor[b][1];
  //     // linearIndices[b] = indices[b][0] + cells[0] * indices[b][1];
  //   }
  // });

  auto sorted = torch::argsort(linearIndices);
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;

  auto sortedIndices = torch::clone(linearIndices);
  auto sortedPositions = torch::clone(points);
  auto sortedSupport = torch::clone(supports);

  auto sort_ = sorted.accessor<int32_t, 1>();
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
  auto sortedIndex_ = sortedIndices.accessor<int32_t, 1>();
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
  auto sortedPosition_ = sortedPositions.accessor<float, 2>();
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
  auto sortedSupport_ = sortedSupport.accessor<float, 1>();
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
  auto points_ = points.accessor<float, 2>();
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
  auto supports_ = supports.accessor<float,1>();
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;

  at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end) {
    for (int32_t b = start; b < end; b++) {
      auto i = sort_[b];
      sortedIndex_[b] = linearIndexAccessor[i];
      sortedPosition_[b][0] = points_[i][0];
      sortedPosition_[b][1] = points_[i][1];
      sortedSupport_[b] = supports_[i];
    }
  });
  // auto b = 0;
  // std::cout << __FILE__ << " " << __LINE__ << ": " << sorted[b] << std::endl;
  // std::cout << __FILE__ << " " << __LINE__ << ": " << points[sort_[b]] << std::endl;
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
  // sortedPositions[sorted] = points;

  // auto sortedIndices = linearIndices[sorted];
  // auto sortedPositions = points[sorted];
  // auto sortedSupport = supports[sorted];
  return {qMin, hMax, cells, sortedPositions, sortedSupport, sortedIndices, sorted};


  // std::cout << "indices: " << indices;


  torch::Tensor z_out = at::empty({points.size(0)}, points.options());

  return {z_out};
}
*/

std::pair<int32_t, int32_t> queryHashMap(int32_t qIDx, int32_t qIDy, at::TensorAccessor<int32_t,2> hashTable, at::TensorAccessor<int32_t,1> cellIndices, at::TensorAccessor<int32_t,1> cumCell,  at::TensorAccessor<int32_t,1> cellSpan, int32_t cellsX, 
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


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::pair<torch::Tensor,torch::Tensor> countNeighborsCUDAImpl(
    torch::Tensor queryParticles_, torch::Tensor support_,
    torch::Tensor sortedParticles, torch::Tensor sortedSupport,
    torch::Tensor hashTable_,
    torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
    torch::Tensor sort_,
    torch::Tensor qMin_,
    float hMax,
    int32_t cellsX, int32_t hashMapLength, int32_t searchRadius);

std::pair<torch::Tensor,torch::Tensor> constructNeighborsCUDAImpl(
    torch::Tensor queryParticles_, torch::Tensor support_,
    torch::Tensor sortedParticles, torch::Tensor sortedSupport,
    torch::Tensor hashTable_,
    torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
    torch::Tensor sort_,
    torch::Tensor qMin_,
    torch::Tensor counters,
    torch::Tensor offsets,
    float hMax,
    int32_t cellsX, int32_t hashMapLength, int32_t searchRadius);

std::vector<torch::Tensor> buildNeighborListCUDA(
    torch::Tensor queryParticles_, torch::Tensor support_,
    torch::Tensor sortedParticles, torch::Tensor sortedSupport,
    torch::Tensor hashTable_,
    torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
    torch::Tensor sort_,
    torch::Tensor qMin_,
    float hMax,
    int32_t cellsX, int32_t hashMapLength, int32_t searchRadius)
{
  CHECK_INPUT(queryParticles_);
  CHECK_INPUT(support_);
  CHECK_INPUT(sortedParticles);
  CHECK_INPUT(sortedSupport);
  CHECK_INPUT(hashTable_);
  CHECK_INPUT(cellIndices_);
  CHECK_INPUT(cumCell_);
  CHECK_INPUT(cellSpan_);
  CHECK_INPUT(sort_);
  CHECK_INPUT(qMin_);

  auto neighCount = countNeighborsCUDAImpl(queryParticles_, support_, sortedParticles, sortedSupport, hashTable_, cellIndices_, cumCell_, cellSpan_, sort_, qMin_, hMax, cellsX, hashMapLength, searchRadius);
  // return {neighCount.first, neighCount.second};

  auto neighborList = constructNeighborsCUDAImpl(queryParticles_, support_, sortedParticles, sortedSupport, hashTable_, cellIndices_, cumCell_, cellSpan_, sort_, qMin_, neighCount.first, neighCount.second, hMax, cellsX, hashMapLength, searchRadius);

  return {neighborList.first, neighborList.second};
}

std::vector<torch::Tensor> buildNeighborList(
    torch::Tensor queryParticles_, torch::Tensor support_,
    torch::Tensor hashTable_,
    torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
    torch::Tensor sort_,
    int32_t cellsX, int32_t hashMapLength, int32_t searchRadius)
{
 
// std::cout << "Number of Threads: " << at::get_num_threads() << std::endl;
// omp_set_num_threads(16);
//   #pragma omp parallel
//   {
//     #pragma omp single
//     printf("num_threads = %d\n", omp_get_num_threads());
//   }
// std::cout << "Number of Threads (OpenMP): " << omp_get_num_threads() << std::endl;

// std::cout << "Number of Threads (OpenMP): " << omp_get_num_threads() << std::endl;
  // omp_set_num_threads(at::get_num_threads() * 2);
    auto queryParticles = queryParticles_.accessor<float, 2>();
    auto support = support_.accessor<float, 1>();
    auto hashTable = hashTable_.accessor<int32_t, 2>();
    auto cumCell = cumCell_.accessor<int32_t, 1>();
    auto cellIndices = cellIndices_.accessor<int32_t, 1>();
    auto cellSpan = cellSpan_.accessor<int32_t, 1>();
    auto sort = sort_.accessor<int32_t, 1>();

    std::mutex m;
    std::vector<std::vector<int32_t>> globalRows;
    std::vector<std::vector<int32_t>> globalCols;
    // std::vector<std::vector<int32_t>> neighborCounters;

    int32_t batch_size = cellIndices.size(0);
    at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end){
      // printf("Hello from thread %d of %d [%lld - %lld]\n", omp_get_thread_num(), omp_get_num_threads(), start, end);
      std::vector<int32_t> rows, cols;
      for (int32_t b = start; b < end; b++) {
        auto cell = cellIndices[b];
        auto qIDx = cell % cellsX;
        auto qIDy = cell / cellsX;
        auto indexPair = queryHashMap(qIDx, qIDy, hashTable, cellIndices, cumCell, cellSpan, cellsX, hashMapLength);
        for (int32_t xx = -searchRadius; xx<= searchRadius; ++xx){
          for (int32_t yy = -searchRadius; yy<= searchRadius; ++yy){
            auto currentIndexPair = queryHashMap(qIDx + xx, qIDy + yy, hashTable, cellIndices, cumCell, cellSpan, cellsX, hashMapLength);
            if(currentIndexPair.first == -1) continue;
            for(int32_t i = indexPair.first; i < indexPair.second; ++i){
              auto xi = queryParticles[i];
              auto hi = support[i];
              for(int32_t j = currentIndexPair.first; j < currentIndexPair.second; ++j){
                auto xj = queryParticles[j];
                auto dist = sqrt((xi[0] - xj[0]) * (xi[0] - xj[0]) + (xi[1] - xj[1]) * (xi[1] - xj[1]));
                if( dist < hi){
                  rows.push_back(sort[i]);
                  cols.push_back(sort[j]);
                }
              }
            }
          }
        }
      }
      if(rows.size() > 0)
      {
    // std::cout << rows.size() << std::endl;
        std::lock_guard<std::mutex> lg(m);
        globalRows.push_back(rows);
        globalCols.push_back(cols);
      } 
    });

    int32_t totalElements = 0;
    for (const auto &v : globalRows)
        totalElements += (int32_t)v.size();
    if (totalElements == 0)
      return {at::empty({0},  torch::TensorOptions().dtype(torch::kInt)), at::empty({0},  torch::TensorOptions().dtype(torch::kInt))};

    // std::cout << totalElements << std::endl;
    
        // return {at::empty({0},  torch::TensorOptions().dtype(torch::kInt)), at::empty({0},  torch::TensorOptions().dtype(torch::kInt))};
    auto rowTensor = at::empty({totalElements}, torch::TensorOptions().dtype(torch::kInt));
    auto colTensor = at::empty({totalElements}, torch::TensorOptions().dtype(torch::kInt));
    std::size_t offset = 0;
    for (std::size_t i = 0; i < globalRows.size(); ++i){
        memcpy(rowTensor.data_ptr() + offset, globalRows[i].data(), globalRows[i].size() * sizeof(int32_t));
        memcpy(colTensor.data_ptr() + offset, globalCols[i].data(), globalCols[i].size() * sizeof(int32_t));
        offset += globalCols[i].size() * sizeof(int32_t);
    }
    return {rowTensor, colTensor};
}

std::vector<torch::Tensor> buildNeighborListUnsortedPerParticle(
    torch::Tensor inputParticles_, torch::Tensor inputSupport_,
    torch::Tensor queryParticles_, torch::Tensor support_,
    torch::Tensor hashTable_,
    torch::Tensor cellIndices_, torch::Tensor cumCell_, torch::Tensor cellSpan_,
    torch::Tensor sort_,
    int32_t cellsX, int32_t hashMapLength, torch::Tensor qMin_, float hMax,int32_t searchRadius)
{
    auto inputParticles = inputParticles_.accessor<float, 2>();
    auto inputSupport = inputSupport_.accessor<float, 1>();
    auto queryParticles = queryParticles_.accessor<float, 2>();
    auto support = support_.accessor<float, 1>();
    auto hashTable = hashTable_.accessor<int32_t, 2>();
    auto cumCell = cumCell_.accessor<int32_t, 1>();
    auto cellIndices = cellIndices_.accessor<int32_t, 1>();
    auto cellSpan = cellSpan_.accessor<int32_t, 1>();
    auto sort = sort_.accessor<int32_t, 1>();
    auto qMin = qMin_.accessor<float, 1>();

    std::mutex m;
    std::vector<std::vector<int32_t>> globalRows;
    std::vector<std::vector<int32_t>> globalCols;

    int32_t batch_size = inputParticles.size(0);
    at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end){
      std::vector<int32_t> rows, cols;
      for (int32_t b = start; b < end; b++) {
        auto xi = inputParticles[b];
        auto hi = inputSupport[b];
        int32_t qIDx = ceil((xi[0] - qMin[0]) / hMax);
        int32_t qIDy = ceil((xi[1] - qMin[1]) / hMax);

        for (int32_t xx = -searchRadius; xx<= searchRadius; ++xx){
          for (int32_t yy = -searchRadius; yy<= searchRadius; ++yy){
            auto currentIndexPair = queryHashMap(qIDx + xx, qIDy + yy, hashTable, cellIndices, cumCell, cellSpan, cellsX, hashMapLength);
            if(currentIndexPair.first == -1) continue;
            for(int32_t j = currentIndexPair.first; j < currentIndexPair.second; ++j){
              auto xj = queryParticles[j];
              auto dist = sqrt((xi[0] - xj[0]) * (xi[0] - xj[0]) + (xi[1] - xj[1]) * (xi[1] - xj[1]));
              if( dist < hi){
                rows.push_back(b);
                cols.push_back(sort[j]);
              }
            }
          }
        }
      }
      if(rows.size() > 0)
      {
        std::lock_guard<std::mutex> lg(m);
        globalRows.push_back(rows);
        globalCols.push_back(cols);
      } 
    });

    int32_t totalElements = 0;
    for (const auto &v : globalRows)
        totalElements += (int32_t)v.size();
    if (totalElements == 0)
      return {at::empty({0},  torch::TensorOptions().dtype(torch::kInt)), at::empty({0},  torch::TensorOptions().dtype(torch::kInt))};

    auto rowTensor = at::empty({totalElements}, torch::TensorOptions().dtype(torch::kInt));
    auto colTensor = at::empty({totalElements}, torch::TensorOptions().dtype(torch::kInt));
    std::size_t offset = 0;
    for (std::size_t i = 0; i < globalRows.size(); ++i){
        memcpy(rowTensor.data_ptr() + offset, globalRows[i].data(), globalRows[i].size() * sizeof(int32_t));
        memcpy(colTensor.data_ptr() + offset, globalCols[i].data(), globalCols[i].size() * sizeof(int32_t));
        offset += globalCols[i].size() * sizeof(int32_t);
    }
    return {rowTensor, colTensor};
}

std::vector<torch::Tensor> buildNeighborListAsymmetric(
    torch::Tensor queryParticlesA_, torch::Tensor supportA_,
    torch::Tensor hashTableA_,
    torch::Tensor cellIndicesA_, torch::Tensor cumCellA_, torch::Tensor cellSpanA_,
    torch::Tensor sortA_,
    torch::Tensor queryParticlesB_, torch::Tensor supportB_,
    torch::Tensor hashTableB_,
    torch::Tensor cellIndicesB_, torch::Tensor cumCellB_, torch::Tensor cellSpanB_,
    torch::Tensor sortB_,
    int32_t cellsX, int32_t hashMapLength)
{
    auto queryParticlesA = queryParticlesA_.accessor<float, 2>();
    auto supportA = supportA_.accessor<float, 1>();
    auto hashTableA = hashTableA_.accessor<int32_t, 2>();
    auto cumCellA = cumCellA_.accessor<int32_t, 1>();
    auto cellIndicesA = cellIndicesA_.accessor<int32_t, 1>();
    auto cellSpanA = cellSpanA_.accessor<int32_t, 1>();
    auto sortA = sortA_.accessor<int32_t, 1>();
    auto queryParticlesB = queryParticlesB_.accessor<float, 2>();
    auto supportB = supportB_.accessor<float, 1>();
    auto hashTableB = hashTableB_.accessor<int32_t, 2>();
    auto cumCellB = cumCellB_.accessor<int32_t, 1>();
    auto cellIndicesB = cellIndicesB_.accessor<int32_t, 1>();
    auto cellSpanB = cellSpanB_.accessor<int32_t, 1>();
    auto sortB = sortB_.accessor<int32_t, 1>();

    std::mutex m;
    std::vector<std::vector<int32_t>> globalRows;
    std::vector<std::vector<int32_t>> globalCols;

    int32_t batch_size = cellIndicesA.size(0);
    at::parallel_for(0, batch_size, 0, [&](int32_t start, int32_t end){
      std::vector<int32_t> rows, cols;
      for (int32_t b = start; b < end; b++) {
        auto cell = cellIndicesA[b];
        auto qIDx = cell % cellsX;
        auto qIDy = cell / cellsX;
        auto indexPair = queryHashMap(qIDx, qIDy, hashTableA, cellIndicesA, cumCellA, cellSpanA, cellsX, hashMapLength);
        for (int32_t xx = -1; xx<= 1; ++xx){
          for (int32_t yy = -1; yy<= 1; ++yy){
            auto currentIndexPair = queryHashMap(qIDx + xx, qIDy + yy, hashTableB, cellIndicesB, cumCellB, cellSpanB, cellsX, hashMapLength);
            if(currentIndexPair.first == -1) continue;
            for(int32_t i = indexPair.first; i < indexPair.second; ++i){
              auto xi = queryParticlesA[i];
              auto hi = supportA[i];
              for(int32_t j = currentIndexPair.first; j < currentIndexPair.second; ++j){
                auto xj = queryParticlesB[j];
                auto dist = sqrt((xi[0] - xj[0]) * (xi[0] - xj[0]) + (xi[1] - xj[1]) * (xi[1] - xj[1]));
                if( dist < hi){
                  rows.push_back(sortA[i]);
                  cols.push_back(sortB[j]);
                }
              }
            }
          }
        }
      }
      {
        std::lock_guard<std::mutex> lg(m);
        globalRows.push_back(rows);
        globalCols.push_back(cols);
      } 
    });

    int32_t totalElements = 0;
    for (const auto &v : globalRows)
        totalElements += (int32_t)v.size();

    auto rowTensor = at::empty({totalElements}, torch::TensorOptions().dtype(torch::kInt));
    auto colTensor = at::empty({totalElements}, torch::TensorOptions().dtype(torch::kInt));
    std::size_t offset = 0;
    for (std::size_t i = 0; i < globalRows.size(); ++i){
        memcpy(rowTensor.data_ptr() + offset, globalRows[i].data(), globalRows[i].size() * sizeof(int32_t));
        memcpy(colTensor.data_ptr() + offset, globalCols[i].data(), globalCols[i].size() * sizeof(int32_t));
        offset += globalCols[i].size() * sizeof(int32_t);
    }
    return {rowTensor, colTensor};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("buildNeighborListCUDA", &buildNeighborListCUDA, "LLTM backward (CUDA)");
  m.def("buildNeighborList", &buildNeighborList, "LLTM backward (CUDA)");
  m.def("buildNeighborListAsymmetric", &buildNeighborListAsymmetric, "LLTM backward (CUDA)");
  m.def("buildNeighborListUnsortedPerParticle", &buildNeighborListUnsortedPerParticle, "LLTM backward (CUDA)");
}