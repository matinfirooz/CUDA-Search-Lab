#include "cudasearch/search.hpp"
#include "cudasearch/batched_core.hpp"
#include "cudasearch/cuda_utils.cuh"
#include <algorithm>
#include <chrono>
#include <climits>
#include <limits>

namespace cs {
namespace {
using detail::Buffer;
struct DeviceGraph {
    int n;
    Buffer<int> row,col;
    Buffer<unsigned int> weight;
    explicit DeviceGraph(const Graph& g):n(g.n),row(g.row),col(g.col),weight(g.weight) {}
    detail::View view(const Cost* h=nullptr) const {
        return {n,row.get(),col.get(),weight.get(),h};
    }
};
__global__ void initialize(Cost* d,int n,int s) {
    for(int v=blockIdx.x*blockDim.x+threadIdx.x;v<n;v+=blockDim.x*gridDim.x)
        d[v]=v==s?0:INF;
}
__global__ void bfs_expand(detail::View g,const int* frontier,int count,
                           int* next,int* next_count,Cost* d,Cost next_depth) {
    for(int i=blockIdx.x;i<count;i+=gridDim.x) {
        int u=frontier[i];
        for(int e=g.row[u]+threadIdx.x;e<g.row[u+1];e+=blockDim.x) {
            int v=g.col[e];
            if(atomicCAS(&d[v],INF,next_depth)==INF) {
                int slot=atomicAdd(next_count,1);
                next[slot]=v; // at most one enqueue per vertex
            }
        }
    }
}
std::vector<Cost> bfs_field(const DeviceGraph& g,int s,Cost& expansions,Cost& rounds) {
    Buffer<Cost> d(g.n); Buffer<int> a(g.n),b(g.n),count(1);
    initialize<<<detail::blocks(g.n),256>>>(d.get(),g.n,s); detail::launch_check();
    CS_CUDA(cudaMemcpy(a.get(),&s,sizeof(int),cudaMemcpyHostToDevice));
    int active=1; Cost depth=0;
    while(active) {
        expansions+=active; ++rounds; count.zero();
        bfs_expand<<<std::min(active,65535),128>>>(g.view(),a.get(),active,b.get(),count.get(),d.get(),++depth);
        detail::launch_check(); active=count.scalar(); std::swap(a,b);
    }
    return d.download();
}
__device__ Cost priority(int u,const Cost* d,const Cost* h,int mode,Cost factor) {
    Cost hu=h?h[u]:0;
    if(mode==2) return hu; // greedy
    if(mode==0) return d[u]; // Dijkstra
    return detail::plus(d[u],hu*factor);
}
__global__ void init_best(Cost* d,int* open,int* parent,int n,int s) {
    for(int v=blockIdx.x*blockDim.x+threadIdx.x;v<n;v+=blockDim.x*gridDim.x) {
        d[v]=v==s?0:INF; open[v]=v==s; parent[v]=v==s?s:-1;
    }
}
__global__ void select_key(const Cost* d,const Cost* h,const int* open,
                           int n,int mode,Cost factor,Cost* best) {
    Cost local=INF;
    for(int v=blockIdx.x*blockDim.x+threadIdx.x;v<n;v+=blockDim.x*gridDim.x)
        if(open[v]) { Cost key=priority(v,d,h,mode,factor); if(key<local) local=key; }
    // Baseline reduction: one atomic per thread, O(V) scan per expansion.
    if(local<INF) atomicMin(best,local);
}
__global__ void select_vertex(const Cost* d,const Cost* h,const int* open,
                              int n,int mode,Cost factor,const Cost* key,int* best) {
    int local=INT_MAX;
    for(int v=blockIdx.x*blockDim.x+threadIdx.x;v<n;v+=blockDim.x*gridDim.x)
        if(open[v] && priority(v,d,h,mode,factor)==*key && v<local) local=v;
    if(local<INT_MAX) atomicMin(best,local);
}
__global__ void close_vertex(int* open,int u,const Cost* d,Cost* source_distance) {
    if(threadIdx.x==0) { open[u]=0; *source_distance=d[u]; }
}
__global__ void relax_selected(detail::View g,int u,const Cost* source_distance,
                                Cost* d,int* open,int* parent) {
    const Cost du=*source_distance;
    for(int e=g.row[u]+blockIdx.x*blockDim.x+threadIdx.x;e<g.row[u+1];
        e+=blockDim.x*gridDim.x) {
        int v=g.col[e]; Cost candidate=detail::plus(du,g.weight[e]);
        Cost old=atomicMin(&d[v],candidate);
        if(candidate<old) {
            // CSR validation guarantees unique destinations within this row;
            // exactly one source is expanded in this launch.
            parent[v]=u; open[v]=1;
        }
    }
}
Result gpu_best(const Graph& host,const DeviceGraph& g,int s,int t,Algorithm a,const Options& o) {
    Buffer<Cost> d(g.n),h(o.heuristic),key(1),source_distance(1);
    Buffer<int> open(g.n),parent(g.n),chosen(1);
    init_best<<<detail::blocks(g.n),256>>>(d.get(),open.get(),parent.get(),g.n,s); detail::launch_check();
    int mode=a==Algorithm::Dijkstra?0:a==Algorithm::Greedy?2:1;
    Cost factor=a==Algorithm::WeightedAStar?o.heuristic_weight:1; Result r;
    while(true) {
        key.scalar(INF); chosen.scalar(INT_MAX);
        select_key<<<detail::blocks(g.n),256>>>(d.get(),h.get(),open.get(),g.n,mode,factor,key.get());
        detail::launch_check();
        select_vertex<<<detail::blocks(g.n),256>>>(d.get(),h.get(),open.get(),g.n,mode,factor,key.get(),chosen.get());
        detail::launch_check(); int u=chosen.scalar();
        if(u==INT_MAX) break;
        if(r.expansions>=o.max_expansions) { r.status=Status::LimitReached; break; }
        ++r.expansions; ++r.iterations;
        close_vertex<<<1,1>>>(open.get(),u,d.get(),source_distance.get()); detail::launch_check();
        if(u==t) {
            auto parents=parent.download();
            for(int v=t,steps=0;;v=parents[v]) {
                if(v<0 || v>=g.n || steps++>=g.n) throw std::logic_error("invalid GPU parent chain");
                r.path.push_back(v); if(v==s) break;
            }
            std::reverse(r.path.begin(),r.path.end());
            r.status=Status::Found; r.cost=path_cost(host,r.path); r.objective=r.cost; break;
        }
        int degree=host.row[u+1]-host.row[u];
        if(degree) {
            relax_selected<<<detail::blocks(degree),256>>>(g.view(),u,source_distance.get(),d.get(),open.get(),parent.get());
            detail::launch_check();
        }
    }
    return r;
}
__global__ void bf_round(detail::View g,const Cost* old,Cost* next,int* changed,Cost* expansions) {
    for(int u=blockIdx.x*blockDim.x+threadIdx.x;u<g.n;u+=blockDim.x*gridDim.x) {
        if(old[u]>=INF) continue;
        atomicAdd(expansions,1ULL);
        for(int e=g.row[u];e<g.row[u+1];++e) {
            Cost c=detail::plus(old[u],g.weight[e]);
            if(c<atomicMin(&next[g.col[e]],c)) atomicExch(changed,1);
        }
    }
}
Result gpu_bf(const Graph& host,const DeviceGraph& g,int s,int t) {
    Buffer<Cost> d(g.n),next(g.n),expanded(1); Buffer<int> changed(1);
    expanded.zero(); initialize<<<detail::blocks(g.n),256>>>(d.get(),g.n,s); detail::launch_check();
    Cost rounds=0;
    for(int k=1;k<g.n;++k) {
        ++rounds; changed.zero();
        CS_CUDA(cudaMemcpy(next.get(),d.get(),static_cast<std::size_t>(g.n)*sizeof(Cost),cudaMemcpyDeviceToDevice));
        bf_round<<<detail::blocks(g.n),256>>>(g.view(),d.get(),next.get(),changed.get(),expanded.get());
        detail::launch_check(); std::swap(d,next);
        if(!changed.scalar()) break;
    }
    auto r=result_from_distances(host,s,t,d.download()); r.expansions=expanded.scalar(); r.iterations=rounds; return r;
}
__global__ void fill_inf(Cost* data,std::size_t n) {
    for(std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<n;
        i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) data[i]=INF;
}
__global__ void bucket_min(const Cost* d,const Cost* processed,int n,Cost delta,Cost* bucket) {
    Cost local=INF;
    for(int v=blockIdx.x*blockDim.x+threadIdx.x;v<n;v+=blockDim.x*gridDim.x)
        if(d[v]<processed[v] && d[v]/delta<local) local=d[v]/delta;
    if(local<INF) atomicMin(bucket,local);
}
__global__ void bucket_mark(const Cost* d,Cost* processed,Cost* snapshot,int* active,
    int* removed,int n,Cost bucket,Cost delta,int* count) {
    for(int v=blockIdx.x*blockDim.x+threadIdx.x;v<n;v+=blockDim.x*gridDim.x) {
        active[v]=(d[v]<processed[v] && d[v]/delta==bucket);
        if(active[v]) {
            processed[v]=d[v]; snapshot[v]=d[v]; removed[v]=1; atomicAdd(count,1);
        }
    }
}
__global__ void bucket_relax(detail::View g,const int* active,const Cost* snapshot,
                             Cost* d,Cost delta,bool light) {
    for(int u=blockIdx.x*blockDim.x+threadIdx.x;u<g.n;u+=blockDim.x*gridDim.x) {
        if(!active[u]) continue;
        Cost du=snapshot[u];
        for(int e=g.row[u];e<g.row[u+1];++e)
            if((static_cast<Cost>(g.weight[e])<=delta)==light)
                atomicMin(&d[g.col[e]],detail::plus(du,g.weight[e]));
    }
}
Result gpu_delta(const Graph& host,const DeviceGraph& g,int s,int t,Cost delta) {
    Buffer<Cost> d(g.n),processed(g.n),snapshot(g.n),bucket(1);
    Buffer<int> active(g.n),removed(g.n),count(1);
    initialize<<<detail::blocks(g.n),256>>>(d.get(),g.n,s); detail::launch_check();
    fill_inf<<<detail::blocks(g.n),256>>>(processed.get(),g.n); detail::launch_check();
    Cost rounds=0,expanded=0;
    while(true) {
        bucket.scalar(INF);
        bucket_min<<<detail::blocks(g.n),256>>>(d.get(),processed.get(),g.n,delta,bucket.get());
        detail::launch_check(); Cost b=bucket.scalar(); if(b==INF) break;
        removed.zero();
        while(true) {
            count.zero();
            bucket_mark<<<detail::blocks(g.n),256>>>(d.get(),processed.get(),snapshot.get(),active.get(),
                removed.get(),g.n,b,delta,count.get());
            detail::launch_check(); int size=count.scalar(); if(!size) break;
            expanded+=size; ++rounds;
            bucket_relax<<<detail::blocks(g.n),256>>>(g.view(),active.get(),snapshot.get(),d.get(),delta,true);
            detail::launch_check();
        }
        // Freeze distances before heavy-edge relaxation as well: source loads
        // never race atomic updates, including updates that do not lower a label.
        CS_CUDA(cudaMemcpy(snapshot.get(),d.get(),static_cast<std::size_t>(g.n)*sizeof(Cost),cudaMemcpyDeviceToDevice));
        bucket_relax<<<detail::blocks(g.n),256>>>(g.view(),removed.get(),snapshot.get(),d.get(),delta,false);
        detail::launch_check();
    }
    auto r=result_from_distances(host,s,t,d.download()); r.expansions=expanded; r.iterations=rounds; return r;
}
__global__ void fw_initialize(detail::View g,Cost* d) {
    std::size_t cells=static_cast<std::size_t>(g.n)*g.n;
    for(std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<cells;
        i+=static_cast<std::size_t>(blockDim.x)*gridDim.x)
        d[i]=(i/g.n==i%g.n)?0:INF;
}
__global__ void fw_edges(detail::View g,Cost* d) {
    for(int u=blockIdx.x*blockDim.x+threadIdx.x;u<g.n;u+=blockDim.x*gridDim.x)
        for(int e=g.row[u];e<g.row[u+1];++e) {
            auto i=static_cast<std::size_t>(u)*g.n+g.col[e];
            if(g.weight[e]<d[i]) d[i]=g.weight[e];
        }
}
__global__ void fw_pivot(const Cost* d,Cost* row,Cost* col,int n,int k) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
        row[i]=d[static_cast<std::size_t>(k)*n+i];
        col[i]=d[static_cast<std::size_t>(i)*n+k];
    }
}
__global__ void fw_update(Cost* d,const Cost* row,const Cost* col,int n) {
    std::size_t cells=static_cast<std::size_t>(n)*n;
    for(std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<cells;
        i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        Cost c=detail::plus(col[i/n],row[i%n]); if(c<d[i]) d[i]=c;
    }
}
Result gpu_fw(const Graph& host,const DeviceGraph& g,int s,int t,const Options& o) {
    std::size_t n=g.n,cells=n*n;
    if(cells+2*n>o.memory_limit_bytes/sizeof(Cost)) throw std::invalid_argument("Floyd-Warshall exceeds memory limit");
    Buffer<Cost> d(cells),row(n),col(n);
    fw_initialize<<<detail::blocks(cells),256>>>(g.view(),d.get()); detail::launch_check();
    fw_edges<<<detail::blocks(n),256>>>(g.view(),d.get()); detail::launch_check();
    for(int k=0;k<g.n;++k) {
        fw_pivot<<<detail::blocks(n),256>>>(d.get(),row.get(),col.get(),g.n,k); detail::launch_check();
        fw_update<<<detail::blocks(cells),256>>>(d.get(),row.get(),col.get(),g.n); detail::launch_check();
    }
    std::vector<Cost> distance(n);
    CS_CUDA(cudaMemcpy(distance.data(),d.get()+static_cast<std::size_t>(s)*n,n*sizeof(Cost),cudaMemcpyDeviceToHost));
    auto r=result_from_distances(host,s,t,distance); r.iterations=n; return r;
}
__global__ void query_batch(detail::View g,const int* starts,const int* goals,
    std::size_t queries,Algorithm a,detail::Params p,const Cost* heuristics,
    int* ints,Cost* costs,int* paths,detail::Raw* results) {
    for(std::size_t q=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;q<queries;
        q+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        std::size_t n=g.n; int* base=ints+q*7*n; Cost* c=costs+q*2*n;
        detail::Scratch w{base,base+n,base+2*n,base+3*n,base+4*n,base+5*n,base+6*n,c,c+n};
        auto view=g; view.heuristic=heuristics+q*n;
        results[q]=detail::run_serial(view,starts[q],goals[q],a,p,w,paths+q*n);
    }
}
double milliseconds(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
}
} // namespace
bool cuda_available() {
    int count=0; cudaError_t e=cudaGetDeviceCount(&count);
    if(e!=cudaSuccess) { cudaGetLastError(); return false; }
    return count>0;
}
std::string cuda_device_name() {
    if(!cuda_available()) return "no usable CUDA device";
    int id=0; CS_CUDA(cudaGetDevice(&id)); cudaDeviceProp p{};
    CS_CUDA(cudaGetDeviceProperties(&p,id)); return p.name;
}
std::vector<Result> search_cuda_batch(const Graph& g,
    const std::vector<std::pair<int,int>>& queries,Algorithm a,const Options& o,
    const std::vector<std::vector<Cost>>& heuristics) {
    auto begin=std::chrono::steady_clock::now();
    g.validate();
    if(!is_batched(a)) throw std::invalid_argument("batch mode supports dfs,dls,iddfs,idastar,beam,hill");
    if(queries.empty()) return {};
    if(!heuristics.empty() && heuristics.size()!=queries.size())
        throw std::invalid_argument("one heuristic array is required per query");
    std::size_t n=g.n,q=queries.size();
    constexpr std::size_t bytes_per_vertex=8*sizeof(int)+3*sizeof(Cost);
    std::size_t bytes_per_query=n*bytes_per_vertex+sizeof(detail::Raw)+2*sizeof(int);
    if(!n || q>std::numeric_limits<std::size_t>::max()/n ||
       q>o.memory_limit_bytes/bytes_per_query)
        throw std::invalid_argument("batch scratch exceeds memory limit; reduce batch size or graph size");
    std::vector<int> starts(q),goals(q); std::vector<Cost> h(q*n,0);
    for(std::size_t i=0;i<q;++i) {
        Options options=o; if(!heuristics.empty()) options.heuristic=heuristics[i];
        validate_request(g,queries[i].first,queries[i].second,a,options);
        starts[i]=queries[i].first; goals[i]=queries[i].second;
        if(!options.heuristic.empty()) std::copy(options.heuristic.begin(),options.heuristic.end(),h.begin()+i*n);
    }
    if(!cuda_available()) throw std::runtime_error("no usable CUDA device");
    DeviceGraph device(g); Buffer<int> ds(starts),dt(goals),ints(q*n*7),paths(q*n);
    paths.zero();
    Buffer<Cost> dh(h),costs(q*n*2); Buffer<detail::Raw> raw(q);
    detail::Params p{o.depth_limit,o.beam_width,o.max_expansions};
    query_batch<<<detail::blocks(q,128),128>>>(device.view(),ds.get(),dt.get(),q,a,p,dh.get(),
        ints.get(),costs.get(),paths.get(),raw.get()); detail::launch_check();
    auto results=raw.download(); auto path=paths.download();
    std::vector<Result> out(q);
    for(std::size_t i=0;i<q;++i) {
        out[i].status=results[i].status; out[i].cost=results[i].cost; out[i].objective=results[i].objective;
        out[i].expansions=results[i].expansions; out[i].iterations=results[i].iterations;
        if(out[i].status==Status::Found)
            out[i].path.assign(path.begin()+i*n,path.begin()+i*n+results[i].length);
    }
    double elapsed=milliseconds(begin); for(auto& r:out) r.elapsed_ms=elapsed;
    return out;
}
Result search_cuda(const Graph& g,int s,int t,Algorithm a,const Options& o) {
    auto begin=std::chrono::steady_clock::now();
    validate_request(g,s,t,a,o);
    if(is_batched(a)) return search_cuda_batch(g,{{s,t}},a,o).front();
    if(!cuda_available()) throw std::runtime_error("no usable CUDA device");
    DeviceGraph device(g); Result r;
    if(a==Algorithm::Bfs) {
        Cost expanded=0,rounds=0; auto d=bfs_field(device,s,expanded,rounds);
        r=result_from_distances(g,s,t,d,true); r.expansions=expanded; r.iterations=rounds;
    } else if(a==Algorithm::BidirectionalBfs) {
        Graph reverse=g.transpose(); DeviceGraph back(reverse);
        Cost expanded=0,rounds=0;
        auto ds=bfs_field(device,s,expanded,rounds),dt=bfs_field(back,t,expanded,rounds);
        Cost best=INF; int meet=-1;
        for(int v=0;v<g.n;++v) if(add(ds[v],dt[v])<best) { best=add(ds[v],dt[v]); meet=v; }
        if(meet>=0) {
            auto left=tight_path(g,s,meet,ds,true),right=tight_path(reverse,t,meet,dt,true);
            std::reverse(right.begin(),right.end()); left.insert(left.end(),right.begin()+1,right.end());
            r.status=Status::Found; r.path=std::move(left); r.cost=path_cost(g,r.path); r.objective=best;
        }
        r.expansions=expanded; r.iterations=rounds;
    } else if(a==Algorithm::BellmanFord) r=gpu_bf(g,device,s,t);
    else if(a==Algorithm::DeltaStepping) r=gpu_delta(g,device,s,t,o.delta);
    else if(a==Algorithm::FloydWarshall) r=gpu_fw(g,device,s,t,o);
    else r=gpu_best(g,device,s,t,a,o);
    CS_CUDA(cudaDeviceSynchronize()); r.elapsed_ms=milliseconds(begin); return r;
}
} // namespace cs
