#include "cudasearch/search.hpp"
#include "cudasearch/batched_core.hpp"
#include <algorithm>
#include <chrono>
#include <functional>
#include <queue>
#include <stdexcept>

namespace cs {
std::vector<Cost> reference_dijkstra(const Graph& g,int s) {
    g.validate();
    if(s<0 || s>=g.n) throw std::invalid_argument("invalid source");
    using Item=std::pair<Cost,int>;
    std::priority_queue<Item,std::vector<Item>,std::greater<Item>> q;
    std::vector<Cost> d(g.n,INF); d[s]=0; q.emplace(0,s);
    while(!q.empty()) {
        auto [du,u]=q.top(); q.pop(); if(du!=d[u]) continue;
        for(int e=g.row[u];e<g.row[u+1];++e) {
            int v=g.col[e]; Cost c=add(du,g.weight[e]);
            if(c<d[v]) { d[v]=c; q.emplace(c,v); }
        }
    }
    return d;
}
std::vector<Cost> reference_bfs(const Graph& g,int s) {
    g.validate();
    if(s<0 || s>=g.n) throw std::invalid_argument("invalid source");
    std::vector<Cost> d(g.n,INF); std::queue<int> q; q.push(s); d[s]=0;
    while(!q.empty()) {
        int u=q.front(); q.pop();
        for(int e=g.row[u];e<g.row[u+1];++e) {
            int v=g.col[e]; if(d[v]==INF) { d[v]=d[u]+1; q.push(v); }
        }
    }
    return d;
}
static Result serial_search(const Graph& g,int s,int t,Algorithm a,const Options& o) {
    std::size_t n=g.n;
    if(n>o.memory_limit_bytes/(8*sizeof(int)+2*sizeof(Cost)))
        throw std::invalid_argument("serial search scratch exceeds memory limit");
    std::vector<int> ints(8*n);
    std::vector<Cost> costs(2*n);
    detail::Scratch w{ints.data(),ints.data()+n,ints.data()+2*n,
        ints.data()+3*n,ints.data()+4*n,ints.data()+5*n,ints.data()+6*n,
        costs.data(),costs.data()+n};
    detail::View v{g.n,g.row.data(),g.col.data(),g.weight.data(),
        o.heuristic.empty()?nullptr:o.heuristic.data()};
    detail::Params p{o.depth_limit,o.beam_width,o.max_expansions};
    int* path=ints.data()+static_cast<std::size_t>(7)*g.n;
    auto raw=detail::run_serial(v,s,t,a,p,w,path);
    Result r; r.status=raw.status; r.cost=raw.cost; r.objective=raw.objective;
    r.expansions=raw.expansions; r.iterations=raw.iterations;
    if(raw.status==Status::Found) r.path.assign(path,path+raw.length);
    return r;
}
static Result best_first(const Graph& g,int s,int t,Algorithm a,const Options& o) {
    using Item=std::pair<Cost,int>;
    std::priority_queue<Item,std::vector<Item>,std::greater<Item>> q;
    std::vector<Cost> d(g.n,INF); std::vector<int> parent(g.n,-1),open(g.n,0);
    auto heuristic=[&](int u){return o.heuristic.empty()?0:o.heuristic[u];};
    auto priority=[&](int u){
        if(a==Algorithm::Greedy) return heuristic(u);
        if(a==Algorithm::Dijkstra) return d[u];
        return add(d[u],heuristic(u)*(a==Algorithm::WeightedAStar?o.heuristic_weight:1));
    };
    d[s]=0; parent[s]=s; open[s]=1; q.emplace(priority(s),s); Result r;
    while(!q.empty()) {
        auto [key,u]=q.top(); q.pop();
        if(!open[u] || key!=priority(u)) continue;
        if(r.expansions>=o.max_expansions) { r.status=Status::LimitReached; break; }
        ++r.expansions; ++r.iterations; open[u]=0;
        if(u==t) {
            // Parent writes are serial; strict improvements and nonnegative
            // weights prevent parent cycles, including zero-weight cycles.
            for(int v=t;;v=parent[v]) { r.path.push_back(v); if(v==s) break; }
            std::reverse(r.path.begin(),r.path.end());
            r.status=Status::Found; r.cost=path_cost(g,r.path); r.objective=r.cost; break;
        }
        for(int e=g.row[u];e<g.row[u+1];++e) {
            int v=g.col[e]; Cost candidate=add(d[u],g.weight[e]);
            if(candidate<d[v]) {
                d[v]=candidate; parent[v]=u; open[v]=1; q.emplace(priority(v),v);
            }
        }
    }
    return r;
}
static Result bidirectional(const Graph& g,int s,int t) {
    // Full two-wave reference. Compute both fields, choose a minimum-hop meet.
    Graph reverse=g.transpose();
    auto ds=reference_bfs(g,s),dt=reference_bfs(reverse,t);
    int meet=-1; Cost best=INF;
    for(int v=0;v<g.n;++v) if(add(ds[v],dt[v])<best) { best=add(ds[v],dt[v]); meet=v; }
    Result r;
    if(meet>=0) {
        auto left=tight_path(g,s,meet,ds,true);
        auto right=tight_path(reverse,t,meet,dt,true);
        std::reverse(right.begin(),right.end()); left.insert(left.end(),right.begin()+1,right.end());
        r.path=std::move(left); r.status=Status::Found; r.cost=path_cost(g,r.path); r.objective=best;
    }
    for(int v=0;v<g.n;++v) r.expansions+=(ds[v]<INF)+(dt[v]<INF);
    return r;
}
Result search_cpu(const Graph& g,int s,int t,Algorithm a,const Options& o) {
    auto begin=std::chrono::steady_clock::now();
    validate_request(g,s,t,a,o); Result r;
    if(is_batched(a)) r=serial_search(g,s,t,a,o);
    else if(a==Algorithm::BidirectionalBfs) r=bidirectional(g,s,t);
    else if(a==Algorithm::Bfs) {
        auto d=reference_bfs(g,s); r=result_from_distances(g,s,t,d,true);
        for(Cost x:d) r.expansions+=(x<INF);
    } else if(a==Algorithm::BellmanFord) {
        std::vector<Cost> d(g.n,INF); d[s]=0;
        for(int round=1;round<g.n;++round) {
            auto next=d; bool changed=false; ++r.iterations;
            for(int u=0;u<g.n;++u) if(d[u]<INF) {
                ++r.expansions;
                for(int e=g.row[u];e<g.row[u+1];++e) {
                    int v=g.col[e]; Cost c=add(d[u],g.weight[e]);
                    if(c<next[v]) { next[v]=c; changed=true; }
                }
            }
            d.swap(next); if(!changed) break;
        }
        auto final=result_from_distances(g,s,t,d); final.expansions=r.expansions;
        final.iterations=r.iterations; r=std::move(final);
    } else if(a==Algorithm::FloydWarshall) {
        std::size_t cells=static_cast<std::size_t>(g.n)*g.n;
        if(cells>o.memory_limit_bytes/sizeof(Cost)) throw std::invalid_argument("Floyd-Warshall exceeds memory limit");
        std::vector<Cost> d(cells,INF);
        for(int u=0;u<g.n;++u) {
            d[static_cast<std::size_t>(u)*g.n+u]=0;
            for(int e=g.row[u];e<g.row[u+1];++e) {
                auto index=static_cast<std::size_t>(u)*g.n+g.col[e];
                d[index]=std::min(d[index],static_cast<Cost>(g.weight[e]));
            }
        }
        for(int k=0;k<g.n;++k) for(int u=0;u<g.n;++u) for(int v=0;v<g.n;++v) {
            auto index=static_cast<std::size_t>(u)*g.n+v;
            d[index]=std::min(d[index],add(d[static_cast<std::size_t>(u)*g.n+k],
                                            d[static_cast<std::size_t>(k)*g.n+v]));
        }
        std::vector<Cost> row(d.begin()+static_cast<std::size_t>(s)*g.n,
                              d.begin()+static_cast<std::size_t>(s+1)*g.n);
        r=result_from_distances(g,s,t,row); r.iterations=g.n;
    } else {
        // The delta-stepping CPU oracle is heap Dijkstra, deliberately independent
        // of the CUDA bucket scheduler rather than a second copy of that code.
        if(a==Algorithm::DeltaStepping) {
            Options oracle_options=o; oracle_options.max_expansions=INF;
            r=best_first(g,s,t,Algorithm::Dijkstra,oracle_options);
        } else r=best_first(g,s,t,a,o);
    }
    r.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    return r;
}
} // namespace cs
