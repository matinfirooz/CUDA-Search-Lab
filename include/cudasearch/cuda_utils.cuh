#pragma once
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cs::detail {
inline void check(cudaError_t code,const char* expression,const char* file,int line) {
    if(code!=cudaSuccess) throw std::runtime_error(std::string(file)+":"+std::to_string(line)+
        " "+expression+": "+cudaGetErrorString(code));
}
#define CS_CUDA(expr) ::cs::detail::check((expr),#expr,__FILE__,__LINE__)
inline void launch_check() { CS_CUDA(cudaGetLastError()); }
inline int blocks(std::size_t n,int threads=256) {
    auto b=(n+threads-1)/threads; return static_cast<int>(b>65535?65535:b?b:1);
}
template<class T> class Buffer {
    T* data_ = nullptr;
    std::size_t size_ = 0;
public:
    explicit Buffer(std::size_t n=0):size_(n) {
        if(n>std::numeric_limits<std::size_t>::max()/sizeof(T)) throw std::overflow_error("device allocation overflow");
        if(n) CS_CUDA(cudaMalloc(reinterpret_cast<void**>(&data_),n*sizeof(T)));
    }
    explicit Buffer(const std::vector<T>& v):Buffer(v.size()) { upload(v); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    Buffer(Buffer&& b) noexcept:data_(b.data_),size_(b.size_) { b.data_=nullptr; b.size_=0; }
    Buffer& operator=(Buffer&& b) noexcept {
        if(this!=&b) { if(data_) cudaFree(data_); data_=b.data_; size_=b.size_; b.data_=nullptr; b.size_=0; }
        return *this;
    }
    ~Buffer() { if(data_) cudaFree(data_); }
    T* get() { return data_; }
    const T* get() const { return data_; }
    std::size_t size() const { return size_; }
    void upload(const std::vector<T>& v) {
        if(v.size()!=size_) throw std::invalid_argument("upload size mismatch");
        if(size_) CS_CUDA(cudaMemcpy(data_,v.data(),size_*sizeof(T),cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> v(size_);
        if(size_) CS_CUDA(cudaMemcpy(v.data(),data_,size_*sizeof(T),cudaMemcpyDeviceToHost));
        return v;
    }
    T scalar() const {
        if(size_!=1) throw std::logic_error("expected scalar buffer");
        T v; CS_CUDA(cudaMemcpy(&v,data_,sizeof(T),cudaMemcpyDeviceToHost)); return v;
    }
    void scalar(T v) {
        if(size_!=1) throw std::logic_error("expected scalar buffer");
        CS_CUDA(cudaMemcpy(data_,&v,sizeof(T),cudaMemcpyHostToDevice));
    }
    void zero() { if(size_) CS_CUDA(cudaMemset(data_,0,size_*sizeof(T))); }
};
} // namespace cs::detail
