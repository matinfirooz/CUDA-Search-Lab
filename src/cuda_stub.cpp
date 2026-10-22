#include "cudasearch/search.hpp"
#include <stdexcept>
namespace cs {
bool cuda_available() { return false; }
std::string cuda_device_name() { return "CUDA backend not compiled"; }
Result search_cuda(const Graph&,int,int,Algorithm,const Options&) {
    throw std::runtime_error("CUDA backend not compiled; configure -DCUDASEARCH_ENABLE_CUDA=ON");
}
std::vector<Result> search_cuda_batch(const Graph&,const std::vector<std::pair<int,int>>&,
    Algorithm,const Options&,const std::vector<std::vector<Cost>>&) {
    throw std::runtime_error("CUDA backend not compiled; configure -DCUDASEARCH_ENABLE_CUDA=ON");
}
}
