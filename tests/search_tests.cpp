#include "cudasearch/search.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
using namespace cs;
std::size_t assertions=0, searches=0;
#define CHECK(expr) do { ++assertions; if(!(expr)) throw std::runtime_error( \
    std::string("line ")+std::to_string(__LINE__)+": "+#expr); } while(false)
void throws(const std::function<void()>& f) {
    bool caught=false; try { f(); } catch(const std::exception&) { caught=true; }
    CHECK(caught);
}
Result run(const Graph& g,int s,int t,Algorithm a,const Options& o,bool gpu) {
    ++searches; return gpu?search_cuda(g,s,t,a,o):search_cpu(g,s,t,a,o);
}
std::vector<Cost> exact_heuristic(const Graph& g,int t) {
    auto h=reference_dijkstra(g.transpose(),t); Cost high=0;
    for(Cost d:h) if(d<INF && d>high) high=d;
    for(auto& d:h) if(d==INF) d=high+1;
    return h;
}
void check_result(const Graph& g,int s,int t,Algorithm a,const Options& o,const Result& r) {
    auto d=reference_dijkstra(g,s),hops=reference_bfs(g,s);
    if(r.status==Status::Found) {
        CHECK(!r.path.empty()); CHECK(r.path.front()==s); CHECK(r.path.back()==t);
        CHECK(r.path.size()<=static_cast<std::size_t>(g.n));
        CHECK(path_cost(g,r.path)==r.cost);
        auto unique=r.path; std::sort(unique.begin(),unique.end());
        CHECK(std::adjacent_find(unique.begin(),unique.end())==unique.end());
        if(is_cost_optimal(a)) CHECK(r.cost==d[t]);
        if(is_hop_optimal(a)) {
            CHECK(r.objective==hops[t]); CHECK(r.objective==r.path.size()-1);
        } else CHECK(r.objective==r.cost);
        if(a==Algorithm::WeightedAStar) CHECK(r.cost<=o.heuristic_weight*d[t]);
        if(a==Algorithm::DepthLimited || a==Algorithm::IterativeDeepening || a==Algorithm::Beam)
            CHECK(r.path.size()-1<=static_cast<std::size_t>(o.depth_limit));
    } else {
        CHECK(r.path.empty()); CHECK(r.cost==INF); CHECK(r.objective==INF);
        if(r.status==Status::Unreachable) CHECK(d[t]==INF);
    }
    // These exact searches must solve a reachable finite instance with a large
    // enough budget/depth, rather than vacuously passing by returning a cutoff.
    if((is_cost_optimal(a) || is_hop_optimal(a) || a==Algorithm::Dfs ||
        a==Algorithm::Greedy || a==Algorithm::WeightedAStar) && d[t]<INF &&
        o.max_expansions>=100000 && o.depth_limit>=g.n-1)
        CHECK(r.status==Status::Found);
}
void fixed_cases(bool gpu) {
    std::vector<std::tuple<int,int,Cost>> chain;
    for(int i=0;i<5;++i) chain.emplace_back(i,i+1,i+1);
    auto g=Graph::from_edges(6,chain); Options o;
    o.depth_limit=6; o.beam_width=6; o.heuristic=exact_heuristic(g,5);
    for(auto a:algorithms()) {
        auto r=run(g,0,5,a,o,gpu); CHECK(r.status==Status::Found); CHECK(r.cost==15);
        check_result(g,0,5,a,o,r);
    }
    auto different=Graph::from_edges(4,{{0,3,10},{0,1,1},{1,2,1},{2,3,1}});
    o.heuristic=exact_heuristic(different,3);
    for(auto a:algorithms()) check_result(different,0,3,a,o,run(different,0,3,a,o,gpu));
    CHECK(run(different,0,3,Algorithm::Bfs,o,gpu).cost==10);
    CHECK(run(different,0,3,Algorithm::AStar,o,gpu).cost==3);
    CHECK(run(different,0,3,Algorithm::Dijkstra,o,gpu).cost==3);

    auto zero=Graph::from_edges(5,{{0,0,0},{0,1,0},{1,0,0},{1,2,0},
        {2,1,0},{2,3,1},{3,4,0},{0,4,9},{0,1,7}});
    CHECK(zero.edge_cost(0,1)==0);
    o.heuristic=exact_heuristic(zero,4);
    for(auto a:algorithms()) check_result(zero,0,4,a,o,run(zero,0,4,a,o,gpu));
    CHECK(run(zero,0,4,Algorithm::Dijkstra,o,gpu).cost==1);
    for(Cost delta:{1ULL,2ULL,4ULL,17ULL}) {
        o.delta=delta;
        check_result(zero,0,4,Algorithm::DeltaStepping,o,run(zero,0,4,Algorithm::DeltaStepping,o,gpu));
    }
    auto disconnected=Graph::from_edges(4,{{0,1,1},{2,3,1}});
    o.heuristic=exact_heuristic(disconnected,3); o.delta=4;
    for(auto a:algorithms()) {
        auto r=run(disconnected,0,3,a,o,gpu); CHECK(r.status!=Status::Found);
        check_result(disconnected,0,3,a,o,r);
    }
    auto singleton=Graph::from_edges(1,{});
    o.heuristic={0};
    for(auto a:algorithms()) {
        auto r=run(singleton,0,0,a,o,gpu); CHECK(r.status==Status::Found); CHECK(r.cost==0);
        check_result(singleton,0,0,a,o,r);
    }
    auto revisit=Graph::from_edges(4,{{0,1,1},{1,2,1},{0,2,1},{2,3,1}});
    o.heuristic.clear(); o.depth_limit=2;
    auto dls=run(revisit,0,3,Algorithm::DepthLimited,o,gpu);
    CHECK(dls.status==Status::Found); CHECK(dls.path==std::vector<int>({0,2,3}));
    o.depth_limit=1;
    CHECK(run(revisit,0,3,Algorithm::DepthLimited,o,gpu).status==Status::LimitReached);
    o.depth_limit=8; o.max_expansions=1;
    for(auto a:{Algorithm::Dfs,Algorithm::DepthLimited,Algorithm::IterativeDeepening,
                Algorithm::IdaStar,Algorithm::AStar,Algorithm::Dijkstra,Algorithm::Beam}) {
        auto r=run(revisit,0,3,a,o,gpu); CHECK(r.status==Status::LimitReached); CHECK(r.expansions<=1);
    }
    o.max_expansions=1000000;
    auto prune=Graph::from_edges(4,{{0,1,1},{0,2,1},{2,3,1}});
    o.heuristic={2,0,1,0}; o.beam_width=1;
    CHECK(run(prune,0,3,Algorithm::Beam,o,gpu).status==Status::LimitReached);
    o.beam_width=4;
    CHECK(run(prune,0,3,Algorithm::Beam,o,gpu).status==Status::Found);
    o.heuristic.clear();
    CHECK(run(prune,0,3,Algorithm::HillClimbing,o,gpu).status==Status::LocalMinimum);
    auto big=Graph::from_edges(4,{{0,1,4000000000ULL},{1,2,4000000000ULL},{2,3,4000000000ULL}});
    for(auto a:{Algorithm::AStar,Algorithm::Dijkstra,Algorithm::DeltaStepping,
                Algorithm::BellmanFord,Algorithm::FloydWarshall}) {
        auto r=run(big,0,3,a,o,gpu); CHECK(r.cost==12000000000ULL); check_result(big,0,3,a,o,r);
    }
    // Many simultaneous frontier claims and equal-cost predecessors.
    std::vector<std::tuple<int,int,Cost>> fan;
    for(int i=1;i<64;++i) { fan.emplace_back(0,i,1); fan.emplace_back(i,64,1); }
    auto diamond=Graph::from_edges(65,fan); o.depth_limit=64; o.beam_width=65;
    for(auto a:{Algorithm::Bfs,Algorithm::BidirectionalBfs,Algorithm::AStar,
                Algorithm::Dijkstra,Algorithm::BellmanFord,Algorithm::DeltaStepping}) {
        auto r=run(diamond,0,64,a,o,gpu); CHECK(r.cost==2); check_result(diamond,0,64,a,o,r);
    }
    std::istringstream in("S99G\n....\n"); auto grid=Grid::read(in);
    o.heuristic=grid.manhattan(grid.goal); o.depth_limit=grid.graph.n;
    for(auto a:algorithms())
        check_result(grid.graph,grid.start,grid.goal,a,o,run(grid.graph,grid.start,grid.goal,a,o,gpu));
}
void random_cases(bool gpu) {
    std::mt19937 rng(1701);
    int count=gpu?10:40;
    for(int sample=0;sample<count;++sample) {
        int n=2+static_cast<int>(rng()%7);
        std::vector<std::tuple<int,int,Cost>> edges;
        for(int u=0;u<n;++u) for(int v=0;v<n;++v)
            if(rng()%100<30) edges.emplace_back(u,v,rng()%9);
        auto g=Graph::from_edges(n,edges);
        for(int query=0;query<(gpu?3:8);++query) {
            int s=static_cast<int>(rng()%n),t=static_cast<int>(rng()%n);
            Options o; o.depth_limit=n; o.beam_width=n; o.delta=1+rng()%11;
            if(query%2) o.heuristic=exact_heuristic(g,t);
            for(auto a:algorithms()) {
                try { check_result(g,s,t,a,o,run(g,s,t,a,o,gpu)); }
                catch(const std::exception& e) {
                    throw std::runtime_error("random sample="+std::to_string(sample)+" "+
                        name(a)+" "+std::to_string(s)+"->"+std::to_string(t)+": "+e.what());
                }
            }
        }
    }
}
void validation_cases() {
    CHECK(parse_algorithm("ucs")==Algorithm::Dijkstra);
    throws([]{parse_algorithm("nonsense");});
    for(const auto& text:{"2 1\n0 1 -1\n","2 2\n0 1 1\n","2 1\n0 2 1\n",
                         "2 1\n0 1 4294967296\n","0 0\n","2 1\n0 1 1 trailing\n"}) {
        throws([&]{std::istringstream in(text); Graph::read(in);});
    }
    for(const auto& text:{"S.G\n..\n","S?G\n","SSG\n","...\n","\n"}) {
        throws([&]{std::istringstream in(text); Grid::read(in);});
    }
    auto g=Graph::from_edges(3,{{0,1,1},{1,2,1}});
    throws([&]{search_cpu(g,-1,2,Algorithm::AStar);});
    Options o; o.heuristic={4,1,0};
    throws([&]{search_cpu(g,0,2,Algorithm::AStar,o);});
    o.heuristic={1,0}; throws([&]{search_cpu(g,0,2,Algorithm::AStar,o);});
    o.heuristic.clear(); o.delta=0;
    throws([&]{search_cpu(g,0,2,Algorithm::DeltaStepping,o);});
    o.delta=1; o.memory_limit_bytes=1;
    throws([&]{search_cpu(g,0,2,Algorithm::FloydWarshall,o);});
    throws([&]{search_cpu(g,0,2,Algorithm::Dfs,o);});
    g.row[1]=99; throws([&]{g.validate();});
    auto random=Grid::random(8,7,0.8,42);
    CHECK(reference_bfs(random.graph,random.start)[random.goal]==13);
}
void batch_cases() {
    auto g=Graph::from_edges(5,{{0,1,1},{1,2,1},{2,3,2},{3,4,3},{1,0,0}});
    // Exercise multiple blocks, mixed goals, zero-length queries, and isolation
    // between per-query scratch segments.
    std::vector<std::pair<int,int>> queries;
    std::vector<std::vector<Cost>> heuristics;
    for(int i=0;i<257;++i) {
        int s=i%5,t=(i*3+2)%5; queries.emplace_back(s,t);
        heuristics.push_back(exact_heuristic(g,t));
    }
    Options o; o.depth_limit=5; o.beam_width=5;
    for(auto a:algorithms()) if(is_batched(a)) {
        auto results=search_cuda_batch(g,queries,a,o,heuristics); searches+=queries.size();
        CHECK(results.size()==queries.size());
        for(std::size_t q=0;q<queries.size();++q) {
            Options qo=o; qo.heuristic=heuristics[q];
            check_result(g,queries[q].first,queries[q].second,a,qo,results[q]);
        }
    }
    CHECK(search_cuda_batch(g,{},Algorithm::Dfs).empty());
    throws([&]{search_cuda_batch(g,queries,Algorithm::AStar);});
    o.memory_limit_bytes=1; throws([&]{search_cuda_batch(g,queries,Algorithm::Dfs,o);});
}
} // namespace
int main(int argc,char** argv) {
    bool gpu=argc==2 && std::string(argv[1])=="--cuda";
    if(gpu && !cs::cuda_available()) { std::cout<<"SKIP: no CUDA device; GPU kernels were not executed\n"; return 77; }
    try {
        validation_cases(); fixed_cases(gpu); random_cases(gpu);
        if(gpu) batch_cases();
        std::cout<<"PASS: "<<(gpu?"CUDA":"CPU")<<" | "<<searches<<" search cases | "
                 <<assertions<<" assertions\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
