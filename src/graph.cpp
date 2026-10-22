#include "cudasearch/graph.hpp"
#include <algorithm>
#include <cmath>
#include <climits>
#include <limits>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>

namespace cs {
std::vector<Algorithm> algorithms() {
    return {Algorithm::Bfs,Algorithm::BidirectionalBfs,Algorithm::Dfs,
        Algorithm::DepthLimited,Algorithm::IterativeDeepening,Algorithm::Dijkstra,
        Algorithm::AStar,Algorithm::WeightedAStar,Algorithm::Greedy,Algorithm::IdaStar,
        Algorithm::Beam,Algorithm::HillClimbing,Algorithm::BellmanFord,
        Algorithm::DeltaStepping,Algorithm::FloydWarshall};
}
std::string name(Algorithm a) {
    switch(a) {
    case Algorithm::Bfs: return "bfs";
    case Algorithm::BidirectionalBfs: return "bidirectional-bfs";
    case Algorithm::Dfs: return "dfs";
    case Algorithm::DepthLimited: return "dls";
    case Algorithm::IterativeDeepening: return "iddfs";
    case Algorithm::Dijkstra: return "dijkstra";
    case Algorithm::AStar: return "astar";
    case Algorithm::WeightedAStar: return "weighted-astar";
    case Algorithm::Greedy: return "greedy";
    case Algorithm::IdaStar: return "idastar";
    case Algorithm::Beam: return "beam";
    case Algorithm::HillClimbing: return "hill";
    case Algorithm::BellmanFord: return "bellman-ford";
    case Algorithm::DeltaStepping: return "delta-stepping";
    case Algorithm::FloydWarshall: return "floyd-warshall";
    }
    throw std::invalid_argument("unknown algorithm enum");
}
std::string name(Status s) {
    switch(s) {
    case Status::Found: return "found";
    case Status::Unreachable: return "unreachable";
    case Status::LimitReached: return "limit-reached";
    case Status::LocalMinimum: return "local-minimum";
    }
    throw std::invalid_argument("unknown status enum");
}
Algorithm parse_algorithm(const std::string& s) {
    if(s=="ucs") return Algorithm::Dijkstra;
    for(auto a:algorithms()) if(name(a)==s) return a;
    throw std::invalid_argument("unknown algorithm: "+s+" (use --list)");
}
bool is_batched(Algorithm a) {
    return a==Algorithm::Dfs || a==Algorithm::DepthLimited ||
        a==Algorithm::IterativeDeepening || a==Algorithm::IdaStar ||
        a==Algorithm::Beam || a==Algorithm::HillClimbing;
}
bool is_hop_optimal(Algorithm a) {
    return a==Algorithm::Bfs || a==Algorithm::BidirectionalBfs ||
        a==Algorithm::IterativeDeepening;
}
bool is_cost_optimal(Algorithm a) {
    return a==Algorithm::Dijkstra || a==Algorithm::AStar || a==Algorithm::IdaStar ||
        a==Algorithm::BellmanFord || a==Algorithm::DeltaStepping ||
        a==Algorithm::FloydWarshall;
}
Graph Graph::from_edges(int count, std::vector<std::tuple<int,int,Cost>> e) {
    if(count<=0 || count>=INT_MAX) throw std::invalid_argument("vertex count must be in [1, INT_MAX-1]");
    if(e.size()>=static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument("CSR uses signed 32-bit edge offsets");
    for(auto [u,v,w]:e)
        if(u<0 || u>=count || v<0 || v>=count || w>UINT_MAX)
            throw std::invalid_argument("invalid edge endpoint or weight (expected uint32)");
    std::sort(e.begin(),e.end());
    Graph g; g.n=count; g.row.assign(static_cast<std::size_t>(count)+1,0);
    // A simple weighted graph: keep only the cheapest parallel edge.
    int last_u=-1,last_v=-1;
    for(auto [u,v,w]:e) {
        if(u==last_u && v==last_v) continue;
        ++g.row[u+1]; g.col.push_back(v); g.weight.push_back(static_cast<unsigned>(w));
        last_u=u; last_v=v;
    }
    for(int i=1;i<=count;++i) g.row[i]+=g.row[i-1];
    return g;
}
Graph Graph::read(std::istream& in) {
    std::string line; long long n64=-1,m64=-1; bool header=false;
    std::vector<std::tuple<int,int,Cost>> edges;
    while(std::getline(in,line)) {
        auto c=line.find('#'); if(c!=std::string::npos) line.resize(c);
        if(line.find_first_not_of(" \t\r")==std::string::npos) continue;
        std::istringstream s(line); std::string extra;
        if(!header) {
            if(!(s>>n64>>m64) || (s>>extra) || n64<=0 || n64>=INT_MAX ||
               m64<0 || m64>=INT_MAX) throw std::invalid_argument("expected header: vertex_count edge_count");
            header=true; continue;
        }
        long long u,v,w;
        if(!(s>>u>>v>>w) || (s>>extra) || u<0 || u>=n64 || v<0 || v>=n64 ||
           w<0 || static_cast<unsigned long long>(w)>UINT_MAX)
            throw std::invalid_argument("expected edge: u v nonnegative_uint32_weight");
        edges.emplace_back(static_cast<int>(u),static_cast<int>(v),static_cast<Cost>(w));
    }
    if(!header || edges.size()!=static_cast<std::size_t>(m64))
        throw std::invalid_argument("edge count does not match header");
    return from_edges(static_cast<int>(n64),std::move(edges));
}
Graph Graph::transpose() const {
    std::vector<std::tuple<int,int,Cost>> e; e.reserve(col.size());
    for(int u=0;u<n;++u) for(int k=row[u];k<row[u+1];++k)
        e.emplace_back(col[k],u,weight[k]);
    return from_edges(n,std::move(e));
}
void Graph::validate() const {
    if(n<=0 || n>=INT_MAX || row.size()!=static_cast<std::size_t>(n)+1 ||
       col.size()!=weight.size() || col.size()>=static_cast<std::size_t>(INT_MAX) ||
       row[0]!=0 || row[n]!=static_cast<int>(col.size()))
        throw std::invalid_argument("malformed CSR graph");
    for(int u=0;u<n;++u) {
        if(row[u]<0 || row[u]>row[u+1] || row[u+1]>row[n])
            throw std::invalid_argument("invalid CSR row offsets");
        int previous=-1;
        for(int e=row[u];e<row[u+1];++e) {
            if(col[e]<0 || col[e]>=n || col[e]<=previous)
                throw std::invalid_argument("CSR destinations must be unique and sorted per row");
            previous=col[e];
        }
    }
}
Cost Graph::edge_cost(int u,int v) const {
    if(u<0 || u>=n || v<0 || v>=n) return INF;
    auto b=col.begin()+row[u], e=col.begin()+row[u+1];
    auto it=std::lower_bound(b,e,v);
    return it==e || *it!=v ? INF : weight[static_cast<std::size_t>(it-col.begin())];
}
Grid Grid::read(std::istream& in) {
    Grid g; std::string line;
    while(std::getline(in,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        if(line.empty()) throw std::invalid_argument("empty grid row");
        if(g.cells.empty()) g.width=static_cast<int>(line.size());
        if(line.size()!=static_cast<std::size_t>(g.width))
            throw std::invalid_argument("grid must be rectangular");
        for(char c:line) if(c!='S' && c!='G' && c!='.' && c!='#' && !(c>='1'&&c<='9'))
            throw std::invalid_argument("grid chars: S G . # 1..9");
        g.cells.push_back(line);
    }
    if(g.cells.empty() || static_cast<std::size_t>(g.width)*g.cells.size()>=INT_MAX)
        throw std::invalid_argument("grid is empty or too large");
    g.height=static_cast<int>(g.cells.size());
    std::vector<std::tuple<int,int,Cost>> edges;
    const int dx[4]={1,0,-1,0},dy[4]={0,1,0,-1};
    for(int y=0;y<g.height;++y) for(int x=0;x<g.width;++x) {
        int u=y*g.width+x; char c=g.cells[y][x];
        if(c=='S') { if(g.start!=-1) throw std::invalid_argument("multiple S cells"); g.start=u; }
        if(c=='G') { if(g.goal!=-1) throw std::invalid_argument("multiple G cells"); g.goal=u; }
        if(c=='#') continue;
        for(int d=0;d<4;++d) {
            int xx=x+dx[d],yy=y+dy[d];
            if(xx<0 || xx>=g.width || yy<0 || yy>=g.height || g.cells[yy][xx]=='#') continue;
            char target=g.cells[yy][xx];
            edges.emplace_back(u,yy*g.width+xx,target>='1'&&target<='9' ? target-'0':1);
        }
    }
    if(g.start<0 || g.goal<0) throw std::invalid_argument("grid requires exactly one S and one G");
    g.graph=Graph::from_edges(g.width*g.height,std::move(edges));
    return g;
}
Grid Grid::random(int w,int h,double p,unsigned seed) {
    if(w<2 || h<2 || static_cast<Cost>(w)*h>2000000 || !std::isfinite(p) || p<0 || p>=1)
        throw std::invalid_argument("random grid: dimensions >=2, <=2M cells, obstacles in [0,1)");
    std::mt19937 rng(seed); std::bernoulli_distribution obstacle(p);
    std::ostringstream out;
    for(int y=0;y<h;++y) {
        for(int x=0;x<w;++x) {
            char c=(y==0 || x==w-1) ? '.' : obstacle(rng)?'#':'.';
            if(x==0 && y==0) c='S';
            if(x==w-1 && y==h-1) c='G';
            out<<c;
        }
        out<<'\n';
    }
    std::istringstream in(out.str()); return read(in);
}
std::vector<Cost> Grid::manhattan(int t) const {
    if(t<0 || t>=graph.n) throw std::invalid_argument("invalid grid goal");
    std::vector<Cost> h(graph.n);
    for(int u=0;u<graph.n;++u)
        h[u]=static_cast<Cost>(std::abs(u%width-t%width)+std::abs(u/width-t/width));
    return h;
}
void validate_request(const Graph& g,int s,int t,Algorithm a,const Options& o) {
    g.validate();
    if(s<0 || s>=g.n || t<0 || t>=g.n) throw std::invalid_argument("query vertex out of range");
    if(o.depth_limit<0 || o.beam_width<=0 || o.heuristic_weight<1 ||
       o.heuristic_weight>1000000 || o.delta==0 || o.delta>=INF || o.max_expansions==0)
        throw std::invalid_argument("invalid search options");
    if(!o.heuristic.empty() && o.heuristic.size()!=static_cast<std::size_t>(g.n))
        throw std::invalid_argument("heuristic size must equal vertex count");
    for(Cost h:o.heuristic) if(h>=INF/1000001)
        throw std::invalid_argument("heuristic too large for safe weighted priorities");
    Cost max_weight=0;
    for(auto w:g.weight) if(w>max_weight) max_weight=w;
    Cost path_bound=max_weight*static_cast<Cost>(g.n-1);
    if(path_bound>=INF/4)
        throw std::invalid_argument("graph path-cost upper bound exceeds numeric range");
    if(a==Algorithm::WeightedAStar)
        for(Cost h:o.heuristic) if(h*o.heuristic_weight>=INF-path_bound)
            throw std::invalid_argument("weighted heuristic priority exceeds numeric range");
    if((a==Algorithm::AStar || a==Algorithm::IdaStar || a==Algorithm::WeightedAStar) &&
       !o.heuristic.empty()) {
        if(o.heuristic[t]!=0) throw std::invalid_argument("heuristic must be zero at goal");
        for(int u=0;u<g.n;++u) for(int e=g.row[u];e<g.row[u+1];++e)
            if(o.heuristic[u]>add(g.weight[e],o.heuristic[g.col[e]]))
                throw std::invalid_argument("A*/IDA*/weighted A* require a consistent heuristic; use h=0");
    }
}
Cost path_cost(const Graph& g,const std::vector<int>& path) {
    if(path.empty()) return INF;
    for(int v:path) if(v<0 || v>=g.n) return INF;
    Cost sum=0;
    for(std::size_t i=1;i<path.size();++i) {
        Cost w=g.edge_cost(path[i-1],path[i]);
        if(w==INF) return INF;
        sum=add(sum,w);
    }
    return sum;
}
std::vector<int> tight_path(const Graph& g,int s,int t,const std::vector<Cost>& d,bool hops) {
    if(d.size()!=static_cast<std::size_t>(g.n) || d[t]>=INF) return {};
    std::vector<int> parent(g.n,-1); parent[s]=s;
    std::queue<int> q; q.push(s);
    while(!q.empty() && parent[t]<0) {
        int u=q.front(); q.pop();
        for(int e=g.row[u];e<g.row[u+1];++e) {
            int v=g.col[e]; Cost w=hops?1:g.weight[e];
            if(parent[v]<0 && d[u]<INF && add(d[u],w)==d[v]) {
                parent[v]=u; q.push(v);
            }
        }
    }
    if(parent[t]<0) throw std::logic_error("distance field has no tight source-to-goal path");
    std::vector<int> path;
    for(int u=t;;u=parent[u]) { path.push_back(u); if(u==s) break; }
    std::reverse(path.begin(),path.end()); return path;
}
Result result_from_distances(const Graph& g,int s,int t,const std::vector<Cost>& d,bool hops) {
    Result r;
    if(d[t]<INF) {
        r.path=tight_path(g,s,t,d,hops); r.status=Status::Found;
        r.objective=d[t]; r.cost=path_cost(g,r.path);
        if(!hops && r.cost!=r.objective) throw std::logic_error("distance/path mismatch");
    }
    return r;
}
} // namespace cs
