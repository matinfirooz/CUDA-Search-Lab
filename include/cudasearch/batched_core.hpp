#pragma once
// Allocation-free routines shared by CPU tests and CUDA query-parallel kernels.
// These searches have serial dependencies inside one query. GPU parallelism is
// across queries, not a claim of parallel DFS within a single query.
#include "types.hpp"
#ifdef __CUDACC__
#define CS_HD __host__ __device__
#else
#define CS_HD
#endif

namespace cs::detail {
struct View {
    int n;
    const int *row, *col;
    const unsigned int* weight;
    const Cost* heuristic;
};
struct Scratch {
    int *stack, *cursor, *seen, *parent, *active, *temporary, *candidate_parent;
    Cost *cost, *candidate;
};
struct Raw {
    Status status = Status::Unreachable;
    Cost cost = INF, objective = INF, expansions = 0, iterations = 0;
    int length = 0;
};
struct Params {
    int depth_limit, beam_width;
    Cost max_expansions;
};
CS_HD inline Cost plus(Cost a,Cost b) {
    return a>=INF || b>=INF || a>INF-b ? INF:a+b;
}
CS_HD inline Cost h(const View& g,int u) { return g.heuristic ? g.heuristic[u]:0; }
CS_HD inline void emit_stack(Raw& r,const Scratch& w,int depth,int* out,bool hops) {
    r.status=Status::Found; r.length=depth+1; r.cost=w.cost[depth];
    r.objective=hops?static_cast<Cost>(depth):r.cost;
    for(int i=0;i<=depth;++i) out[i]=w.stack[i];
}
// mode 0: graph DFS; mode 1: depth-limited simple paths; mode 2: IDA* contour.
CS_HD inline Status depth_contour(const View& g,int s,int t,const Params& p,
    const Scratch& w,int limit,int mode,Cost threshold,Cost& next_threshold,
    Raw& result,int* out,bool hops) {
    for(int i=0;i<g.n;++i) w.seen[i]=0;
    next_threshold=INF;
    if(mode==2 && h(g,s)>threshold) { next_threshold=h(g,s); return Status::LimitReached; }
    if(result.expansions>=p.max_expansions) return Status::LimitReached;
    int depth=0; bool cut=false;
    w.stack[0]=s; w.cursor[0]=g.row[s]; w.cost[0]=0; w.seen[s]=1;
    ++result.expansions;
    while(depth>=0) {
        int u=w.stack[depth];
        if(u==t) { emit_stack(result,w,depth,out,hops); return Status::Found; }
        if(depth==limit) {
            for(int e=w.cursor[depth];e<g.row[u+1];++e)
                if(!w.seen[g.col[e]]) { cut=true; break; }
            if(mode!=0) w.seen[u]=0;
            --depth; continue;
        }
        bool pushed=false;
        while(w.cursor[depth]<g.row[u+1]) {
            int e=w.cursor[depth]++, v=g.col[e];
            if(w.seen[v]) continue;
            Cost next_g=plus(w.cost[depth],g.weight[e]);
            Cost f=plus(next_g,h(g,v));
            if(mode==2 && f>threshold) {
                if(f<next_threshold) next_threshold=f;
                continue;
            }
            if(result.expansions>=p.max_expansions) return Status::LimitReached;
            ++depth; w.stack[depth]=v; w.cursor[depth]=g.row[v];
            w.cost[depth]=next_g; w.seen[v]=1; ++result.expansions;
            pushed=true; break;
        }
        if(!pushed) { if(mode!=0) w.seen[u]=0; --depth; }
    }
    return cut || (mode==2 && next_threshold<INF) ? Status::LimitReached:Status::Unreachable;
}
CS_HD inline void emit_parents(const View& g,int s,int t,const Scratch& w,
                               Raw& r,int* out) {
    int length=0,u=t;
    while(length<g.n) {
        out[length++]=u; if(u==s) break; u=w.parent[u];
    }
    if(out[length-1]!=s) { r.status=Status::LimitReached; return; }
    for(int i=0;i<length/2;++i) { int v=out[i]; out[i]=out[length-1-i]; out[length-1-i]=v; }
    r.status=Status::Found; r.cost=w.cost[t]; r.objective=r.cost; r.length=length;
}
CS_HD inline Raw run_serial(const View& g,int s,int t,Algorithm a,
                             const Params& p,const Scratch& w,int* out) {
    Raw r;
    if(a==Algorithm::Dfs || a==Algorithm::DepthLimited ||
       a==Algorithm::IterativeDeepening || a==Algorithm::IdaStar) {
        int max_depth=p.depth_limit<g.n-1?p.depth_limit:g.n-1;
        if(a==Algorithm::Dfs) max_depth=g.n-1;
        if(a==Algorithm::IdaStar) {
            Cost threshold=h(g,s);
            while(true) {
                ++r.iterations; Cost next=INF;
                auto state=depth_contour(g,s,t,p,w,g.n-1,2,threshold,next,r,out,false);
                if(state==Status::Found) return r;
                if(r.expansions>=p.max_expansions) { r.status=Status::LimitReached; return r; }
                if(next==INF) { r.status=Status::Unreachable; return r; }
                threshold=next;
            }
        }
        int first=a==Algorithm::IterativeDeepening?0:max_depth;
        for(int limit=first;limit<=max_depth;++limit) {
            ++r.iterations; Cost unused=INF;
            auto state=depth_contour(g,s,t,p,w,limit,a==Algorithm::Dfs?0:1,INF,
                unused,r,out,a==Algorithm::IterativeDeepening);
            if(state==Status::Found) return r;
            r.status=state;
            if(state==Status::Unreachable || r.expansions>=p.max_expansions) return r;
        }
        return r;
    }
    for(int i=0;i<g.n;++i) { w.seen[i]=0; w.parent[i]=-1; w.cost[i]=INF; }
    w.seen[s]=1; w.parent[s]=s; w.cost[s]=0;
    if(a==Algorithm::HillClimbing) {
        int u=s;
        while(true) {
            if(r.expansions>=p.max_expansions) { r.status=Status::LimitReached; return r; }
            ++r.expansions; ++r.iterations;
            if(u==t) { emit_parents(g,s,t,w,r,out); return r; }
            int best=-1; Cost best_h=h(g,u),edge_cost=0;
            for(int e=g.row[u];e<g.row[u+1];++e) {
                int v=g.col[e]; Cost hv=h(g,v);
                if(hv<best_h || (hv==best_h && best>=0 && v<best)) {
                    best=v; best_h=hv; edge_cost=g.weight[e];
                }
            }
            if(best<0) { r.status=Status::LocalMinimum; return r; }
            w.parent[best]=u; w.cost[best]=plus(w.cost[u],edge_cost); u=best;
        }
    }
    // Layer beam search, ranked by g+h. Selected vertices get immutable parents.
    // Pruned candidates can reappear in later layers; selected vertices cannot.
    int count=1; w.active[0]=s; bool pruned=false;
    for(int depth=0;;++depth) {
        ++r.iterations;
        for(int j=0;j<count;++j) if(w.active[j]==t) {
            emit_parents(g,s,t,w,r,out); return r;
        }
        if(depth>=p.depth_limit) { r.status=Status::LimitReached; return r; }
        for(int i=0;i<g.n;++i) w.candidate[i]=INF;
        for(int j=0;j<count;++j) {
            if(r.expansions>=p.max_expansions) { r.status=Status::LimitReached; return r; }
            int u=w.active[j]; ++r.expansions;
            for(int e=g.row[u];e<g.row[u+1];++e) {
                int v=g.col[e]; Cost c=plus(w.cost[u],g.weight[e]);
                if(!w.seen[v] && c<INF && (c<w.candidate[v] ||
                   (c==w.candidate[v] && u<w.candidate_parent[v]))) {
                    w.candidate[v]=c; w.candidate_parent[v]=u;
                }
            }
        }
        int next_count=0;
        for(int k=0;k<p.beam_width && k<g.n;++k) {
            int best=-1; Cost key=INF;
            for(int v=0;v<g.n;++v) if(w.candidate[v]<INF) {
                Cost f=plus(w.candidate[v],h(g,v));
                if(f<key || (f==key && (best<0 || v<best))) { key=f; best=v; }
            }
            if(best<0) break;
            w.temporary[next_count++]=best; w.seen[best]=1;
            w.parent[best]=w.candidate_parent[best]; w.cost[best]=w.candidate[best];
            w.candidate[best]=INF;
        }
        for(int i=0;i<g.n;++i) if(w.candidate[i]<INF) pruned=true;
        if(next_count==0) { r.status=pruned?Status::LimitReached:Status::Unreachable; return r; }
        count=next_count;
        for(int j=0;j<count;++j) w.active[j]=w.temporary[j];
    }
}
} // namespace cs::detail
#undef CS_HD
