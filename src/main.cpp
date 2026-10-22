#include "cudasearch/search.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
using namespace cs;
const char* help=R"(CudaSearchLab - classical graph search in CUDA C++17
Usage:
  searchlab --backend cpu --algo astar --grid examples/maze.txt --validate
  searchlab --backend cuda --algo all --graph examples/weighted.edgelist
  searchlab --backend cuda --algo idastar --grid examples/maze.txt --queries examples/queries.txt
Input: --grid FILE | --graph FILE | --random-grid WIDTH HEIGHT (default 16x16)
Options:
  --start ID --goal ID       Override grid endpoints / graph defaults 0 and V-1
  --queries FILE            Lines: source goal. CUDA batches serial search families.
  --depth N --beam N         DLS/IDDFS/beam depth bound; beam width
  --weight N --delta N       Weighted-A* integer factor; delta bucket width
  --budget N                Expansion budget for best-first and batched searches
  --memory-mib N            Floyd matrix / CUDA batch scratch cap, default 512
  --obstacles P --seed N    Random grid configuration; top/right corridor is carved
  --repeat N --warmup N     Median of repeated complete calls; warmup excluded
  --validate               Compare paths/objectives against independent CPU oracles
  --json FILE --csv FILE    Machine-readable results (full paths in JSON)
  --list --help
Algorithms: bfs bidirectional-bfs dfs dls iddfs dijkstra (alias ucs) astar
            weighted-astar greedy idastar beam hill bellman-ford delta-stepping
            floyd-warshall
)";
long long integer(const std::string& s) {
    std::size_t pos=0; long long v=std::stoll(s,&pos);
    if(pos!=s.size()) throw std::invalid_argument("invalid integer: "+s);
    return v;
}
int small_integer(const std::string& s) {
    auto x=integer(s);
    if(x<0 || x>2000000000) throw std::invalid_argument("integer out of range: "+s);
    return static_cast<int>(x);
}
Cost positive(const std::string& s) {
    auto x=integer(s); if(x<=0) throw std::invalid_argument("expected positive integer: "+s);
    return static_cast<Cost>(x);
}
std::string escaped(const std::string& s) {
    std::string out;
    for(char c:s) {
        if(c=='"' || c=='\\') out+='\\';
        if(c=='\n') out+="\\n"; else out+=c;
    }
    return out;
}
void validate(const Graph& g,int s,int t,Algorithm a,const Options& o,const Result& r) {
    auto costs=reference_dijkstra(g,s),hops=reference_bfs(g,s);
    if(r.status==Status::Found) {
        if(r.path.empty() || r.path.front()!=s || r.path.back()!=t || path_cost(g,r.path)!=r.cost)
            throw std::logic_error("validation failed: invalid path or cost");
        if(is_cost_optimal(a) && r.cost!=costs[t])
            throw std::logic_error("validation failed: nonoptimal cost");
        if(is_hop_optimal(a) && r.objective!=hops[t])
            throw std::logic_error("validation failed: nonoptimal hop count");
        if(a==Algorithm::WeightedAStar && r.cost>costs[t]*o.heuristic_weight)
            throw std::logic_error("validation failed: weighted A* bound");
    } else if(r.status==Status::Unreachable && costs[t]!=INF)
        throw std::logic_error("validation failed: reachable target reported unreachable");
}
struct Row { Algorithm a; int s,t; std::size_t query,batch; Result result; };
void json(std::ostream& out,const std::vector<Row>& rows,const std::string& backend) {
    out<<"[\n"; bool first=true;
    for(const auto& row:rows) {
        if(!first) out<<",\n";
        first=false; const auto& r=row.result;
        out<<"  {\"algorithm\":\""<<name(row.a)<<"\",\"backend\":\""<<escaped(backend)
           <<"\",\"source\":"<<row.s<<",\"goal\":"<<row.t<<",\"query\":"<<row.query
           <<",\"batch_size\":"<<row.batch<<",\"timing_scope\":\""<<(row.batch>1?"batch":"query")
           <<"\",\"status\":\""<<name(r.status)<<"\",\"cost\":";
        if(r.cost==INF) out<<"null"; else out<<r.cost;
        out<<",\"objective\":"; if(r.objective==INF) out<<"null"; else out<<r.objective;
        out<<",\"expansions\":"<<r.expansions<<",\"iterations\":"<<r.iterations
           <<",\"median_end_to_end_ms\":"<<std::setprecision(9)<<r.elapsed_ms<<",\"path\":[";
        for(std::size_t i=0;i<r.path.size();++i) { if(i) out<<','; out<<r.path[i]; }
        out<<"]}";
    }
    out<<"\n]\n";
}
void csv(std::ostream& out,const std::vector<Row>& rows,const std::string& backend) {
    out<<"algorithm,backend,source,goal,query,batch_size,timing_scope,status,cost,objective,expansions,iterations,median_end_to_end_ms\n";
    for(const auto& row:rows) {
        const auto& r=row.result;
        out<<name(row.a)<<','<<backend<<','<<row.s<<','<<row.t<<','<<row.query<<','<<row.batch
           <<','<<(row.batch>1?"batch":"query")<<','<<name(r.status)<<',';
        if(r.cost<INF) out<<r.cost;
        out<<','; if(r.objective<INF) out<<r.objective;
        out<<','<<r.expansions<<','<<r.iterations<<','<<std::setprecision(9)<<r.elapsed_ms<<'\n';
    }
}
} // namespace
int main(int argc,char** argv) {
    using namespace cs;
    try {
        std::string backend="cpu",algo="astar",grid_file,graph_file,json_file,csv_file,queries_file;
        int s=-1,t=-1,width=16,height=16,repeats=1,warmup=0; unsigned seed=42;
        double obstacle=0.2; bool verify=false,random_input=false; Options o;
        auto value=[&](int& i) { if(++i>=argc) throw std::invalid_argument("missing option value"); return std::string(argv[i]); };
        for(int i=1;i<argc;++i) {
            std::string flag=argv[i];
            if(flag=="--help") { std::cout<<help; return 0; }
            if(flag=="--list") { for(auto a:algorithms()) std::cout<<name(a)<<'\n'; std::cout<<"ucs (Dijkstra alias)\n"; return 0; }
            if(flag=="--backend") backend=value(i);
            else if(flag=="--algo") algo=value(i);
            else if(flag=="--grid") grid_file=value(i);
            else if(flag=="--graph") graph_file=value(i);
            else if(flag=="--json") json_file=value(i);
            else if(flag=="--csv") csv_file=value(i);
            else if(flag=="--queries") queries_file=value(i);
            else if(flag=="--start") s=small_integer(value(i));
            else if(flag=="--goal") t=small_integer(value(i));
            else if(flag=="--depth") o.depth_limit=small_integer(value(i));
            else if(flag=="--beam") o.beam_width=small_integer(value(i));
            else if(flag=="--weight") o.heuristic_weight=positive(value(i));
            else if(flag=="--delta") o.delta=positive(value(i));
            else if(flag=="--budget") o.max_expansions=positive(value(i));
            else if(flag=="--memory-mib") {
                Cost mib=positive(value(i)); if(mib>1048576) throw std::invalid_argument("memory limit too large");
                o.memory_limit_bytes=static_cast<std::size_t>(mib)*1024*1024;
            } else if(flag=="--repeat") repeats=small_integer(value(i));
            else if(flag=="--warmup") warmup=small_integer(value(i));
            else if(flag=="--seed") seed=static_cast<unsigned>(small_integer(value(i)));
            else if(flag=="--obstacles") {
                auto v=value(i); std::size_t p=0; obstacle=std::stod(v,&p);
                if(p!=v.size()) throw std::invalid_argument("invalid obstacle fraction");
            } else if(flag=="--random-grid") { width=small_integer(value(i)); height=small_integer(value(i)); random_input=true; }
            else if(flag=="--validate") verify=true;
            else throw std::invalid_argument("unknown option: "+flag);
        }
        if(backend!="cpu" && backend!="cuda") throw std::invalid_argument("backend must be cpu or cuda");
        if(repeats<1 || repeats>10000 || warmup>10000) throw std::invalid_argument("repeat/warmup out of range");
        if(static_cast<int>(!grid_file.empty())+static_cast<int>(!graph_file.empty())+random_input>1)
            throw std::invalid_argument("choose one input mode");
        Graph g; Grid grid; bool use_grid=graph_file.empty();
        if(!graph_file.empty()) {
            std::ifstream file(graph_file); if(!file) throw std::runtime_error("cannot open "+graph_file);
            g=Graph::read(file); if(s<0) s=0; if(t<0) t=g.n-1;
        } else {
            if(!grid_file.empty()) {
                std::ifstream file(grid_file); if(!file) throw std::runtime_error("cannot open "+grid_file);
                grid=Grid::read(file);
            } else grid=Grid::random(width,height,obstacle,seed);
            g=grid.graph; if(s<0) s=grid.start; if(t<0) t=grid.goal;
        }
        std::vector<std::pair<int,int>> queries{{s,t}};
        if(!queries_file.empty()) {
            queries.clear(); std::ifstream file(queries_file); std::string line;
            if(!file) throw std::runtime_error("cannot open "+queries_file);
            while(std::getline(file,line)) {
                auto hash=line.find('#'); if(hash!=std::string::npos) line.resize(hash);
                if(line.find_first_not_of(" \t\r")==std::string::npos) continue;
                std::istringstream stream(line); long long u,v; std::string extra;
                if(!(stream>>u>>v) || (stream>>extra) || u<0 || u>=g.n || v<0 || v>=g.n)
                    throw std::invalid_argument("query line requires two valid vertex IDs");
                queries.emplace_back(static_cast<int>(u),static_cast<int>(v));
            }
            if(queries.empty()) throw std::invalid_argument("query file is empty");
        }
        auto selected=algo=="all"?algorithms():std::vector<Algorithm>{parse_algorithm(algo)};
        std::vector<std::vector<Cost>> heuristics;
        for(auto [u,v]:queries) {
            if(u<0 || u>=g.n || v<0 || v>=g.n) throw std::invalid_argument("invalid query");
            if(use_grid && (grid.cells[u/grid.width][u%grid.width]=='#' ||
                            grid.cells[v/grid.width][v%grid.width]=='#'))
                throw std::invalid_argument("grid query endpoint is a wall");
            heuristics.push_back(use_grid?grid.manhattan(v):std::vector<Cost>{});
        }
        std::cout<<"CudaSearchLab | "<<backend<<" | V="<<g.n<<" E="<<g.col.size()
                 <<" | queries="<<queries.size()<<'\n';
        if(backend=="cuda") {
            if(!cuda_available()) throw std::runtime_error("no usable CUDA backend/device");
            std::cout<<"Device: "<<cuda_device_name()<<'\n';
        }
        std::vector<Row> rows;
        for(auto a:selected) {
            std::vector<Result> results; std::vector<double> times;
            for(int trial=-warmup;trial<repeats;++trial) {
                auto begin=std::chrono::steady_clock::now();
                if(backend=="cuda" && is_batched(a)) results=search_cuda_batch(g,queries,a,o,heuristics);
                else {
                    results.clear();
                    for(std::size_t q=0;q<queries.size();++q) {
                        Options qo=o; qo.heuristic=heuristics[q];
                        auto [u,v]=queries[q];
                        results.push_back(backend=="cpu"?search_cpu(g,u,v,a,qo):search_cuda(g,u,v,a,qo));
                    }
                }
                double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
                if(trial>=0) times.push_back(ms);
            }
            std::sort(times.begin(),times.end());
            double median=times[times.size()/2];
            if(times.size()%2==0) median=(median+times[times.size()/2-1])/2;
            for(std::size_t q=0;q<queries.size();++q) {
                auto [u,v]=queries[q]; auto& r=results[q]; r.elapsed_ms=median;
                Options qo=o; qo.heuristic=heuristics[q];
                if(verify) validate(g,u,v,a,qo,r);
                rows.push_back({a,u,v,q,queries.size(),r});
                std::cout<<std::left<<std::setw(19)<<name(a)<<" q="<<q<<' '<<std::setw(14)<<name(r.status)
                         <<" cost="<<(r.cost==INF?"-":std::to_string(r.cost))
                         <<" objective="<<(r.objective==INF?"-":std::to_string(r.objective))
                         <<" expanded="<<r.expansions<<" median_"<<(queries.size()>1?"batch":"query")
                         <<"_ms="<<std::fixed<<std::setprecision(4)<<median<<(verify?" verified":"")<<'\n';
            }
        }
        if(!json_file.empty()) {
            std::ofstream out(json_file); if(!out) throw std::runtime_error("cannot write "+json_file);
            json(out,rows,backend); if(!out) throw std::runtime_error("failed writing "+json_file);
        }
        if(!csv_file.empty()) {
            std::ofstream out(csv_file); if(!out) throw std::runtime_error("cannot write "+csv_file);
            csv(out,rows,backend); if(!out) throw std::runtime_error("failed writing "+csv_file);
        }
        return 0;
    } catch(const std::exception& e) { std::cerr<<"error: "<<e.what()<<'\n'; return 1; }
}
