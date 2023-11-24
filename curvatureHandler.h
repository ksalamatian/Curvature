#ifndef CURVATURE_CURVATUREHANDLER_H
#define CURVATURE_CURVATUREHANDLER_H
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphml.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/copy.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <atomic>
#include <queue>
#include <thread>
#include <string>
#include "BlockingCollection.h"
#include "emd.h"
#include "GraphSpecial.h"
#define ALPHA 0.98
using namespace boost;
using namespace code_machina;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::microseconds;


class DistanceCache{
private:
    float *distanceMat= nullptr;
    int size;
public:
    explicit DistanceCache(int numVertices):size(numVertices*numVertices){
//        distanceMat=(float *)calloc(size, sizeof(float));
    }
    ~DistanceCache(){
        if (distanceMat != nullptr)
            delete distanceMat;
    }

};

struct GraphType : public GraphSpecial{

    double avgCurv=0.0;
    double stdCurv=0.0;
};

class VertexType:public VertexSpecial {
//    double potential;
public:
    bool active=true;
};

class EdgeType:public EdgeSpecial {
public:
    double dist=1.0;
    int weight= 1;
    double distance=1.0;
    float edist=1.0;
    double ot=std::numeric_limits<double>::infinity();
    bool visited=false;
    double curv=std::numeric_limits<double>::infinity();
    bool active=true;
    bool surgery=false;
};

typedef  adjacency_list <vecS, vecS, undirectedS, VertexType, EdgeType, GraphType > Graph_t;

//typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::undirectedS, Vertex_Generic, Edge_Generic, Graph_info > Graph_t;
typedef graph_traits<Graph_t>::vertex_descriptor Vertex;
typedef graph_traits<Graph_t>::edge_descriptor Edge;
typedef graph_traits<Graph_t>::adjacency_iterator  AdjacencyIterator;
typedef graph_traits<Graph_t>::vertex_iterator VertexIterator;
typedef graph_traits<Graph_t>::edge_iterator EdgeIterator;




struct Perf{
public:
    unsigned long maxQueue=0.0;
    unsigned long counter = 0.0;
    double cacheHitRate=0.0;
    long sourceSize=0.0;
    long destSize=0.0;
};

class GraphNode{
public:
    uint  v, source;
    double dist=std::numeric_limits<double>::infinity();
    bool settled=false;

    GraphNode(uint v, uint source, double dist ): v(v), source(source), dist(dist){}

    void fill(uint vv, uint ssource, double ddist){
        v=vv;
        source=ssource;
        dist=ddist;
    }

    void empty(){
        v=0;
        source=0;
        dist=std::numeric_limits<double>::infinity();
        settled=false;
    }

    bool update_distance(double d) {
        if (d < dist) {
            dist = d;
            return true;
        } else
            return false;
    }
};


class GraphNodeArray{
public:
    unsigned long graphSize;
    vector<Vertex> &sources;
    double *nodeArray;
//    double *minDist;
    uint *settledCount;
    uint *minSource;
    DistanceCache &distanceCache;
    GraphNodeArray(unsigned long graphSize, vector<Vertex> &sources, DistanceCache &distanceCache): graphSize(graphSize), sources(sources),
                                                                                          distanceCache(distanceCache){
        nodeArray = new double[graphSize*(sources.size()+1)];
        for (unsigned int i=0; graphSize * (sources.size() + 1) > i; nodeArray[i++]=std::numeric_limits<double>::infinity());
//        minDist=&(nodeArray[graphSize*sources.size()]);
        minSource= new unsigned int[graphSize];
        settledCount=new uint[graphSize];
        for (int i=0;i<graphSize;settledCount[i++]=0);
    };

    ~GraphNodeArray(){
        delete[] nodeArray;
        delete[] minSource;
        delete[] settledCount;
    }

    double get(uint vind, uint sourceind) const{
        return nodeArray[sourceind*graphSize+vind];
    };

    bool checkSet(uint vind, uint sourceind, double dist) const {
        if (nodeArray[sourceind*graphSize+vind]>=dist){
            nodeArray[sourceind*graphSize+vind]=dist;
            return true;
        }
        return false;
    }

    bool isSettled(uint vind, uint sourceind) const {
        // Assume that a negative or 0 distance means that it is settled
        if (get(vind, sourceind)>0.0) {
            return false;
        } else {
            return true;
        }
    }

    bool settle(uint vind, uint sourceind) const {
        nodeArray[sourceind*graphSize+vind]=-nodeArray[sourceind*graphSize+vind];
        settledCount[vind]++;
//        distanceCache.set(sourceind,vind,);
        return true;
    }

    bool isFinished(uint vind) const{
        return sources.size() <= settledCount[vind];
    }

};

class GraphNodeFactory{
public:
    GraphNodeFactory()= default;
    ~GraphNodeFactory(){
        GraphNode *elem;
        while (!graphNodeQueue.empty()){
            delete graphNodeQueue.front();
            graphNodeQueue.pop();
        }
    }

    GraphNode *get(uint v, uint source, double dist){
        GraphNode *elem;
        if (graphNodeQueue.empty()){
            elem=new GraphNode(v,source,dist);
            numb++;
        } else {
            elem = graphNodeQueue.front();
            elem->fill(v,source, dist);
            graphNodeQueue.pop();
        }
        return elem;
    };

    void yield(GraphNode *graphNode){
        graphNode->empty();
        graphNodeQueue.push(graphNode);
        yi++;
    }

    long number() const{
        return numb;
    }

private:
    std::queue<GraphNode*> graphNodeQueue;
    long numb=0, yi=0;

};

struct myComp {
    constexpr bool operator()(
            GraphNode* const& a,
            GraphNode* const& b)
    const noexcept
    {
        return a->dist > b->dist;
    }
};

Perf SSSD_search(vector<vector<double>> *ddists, int offset, int QUANTA, vector<Vertex> &ssources,
                 vector<Vertex> &dests, Graph_t &g, DistanceCache &distanceCache, GraphNodeFactory &graphNodeFact) {
    vector<Vertex> sources(ssources.begin() + offset, ssources.begin() + offset + QUANTA);
    Perf perf;
    perf.sourceSize = sources.size();
    perf.destSize = dests.size();
    class my_visitor : default_bfs_visitor {
    protected:
        Vertex dest;
    public:
        my_visitor(Vertex dest)
                : dest(dest) {};

        void initialize_vertex(const Vertex &s, const Graph_t &g) const {}

        void discover_vertex(const Vertex &s, const Graph_t &g) const {}

        void examine_vertex(const Vertex &s, const Graph_t &g) const {}

        void examine_edge(const Edge &e, const Graph_t &g) const {}

        void edge_relaxed(const Edge &e, const Graph_t &g) const {}

        void edge_not_relaxed(const Edge &e, const Graph_t &g) const {}

        void finish_vertex(const Vertex &s, const Graph_t &g) const {
            if (dest == s)
                throw (2);
        }
    };
    int i=0,j=0;
    for (Vertex src: sources) {
        j=0;
        for (Vertex dst: dests) {
            my_visitor vis(dst);
            std::vector<Vertex> p(num_vertices(g));
            std::vector<double> d(num_vertices(g));
            try{
                dijkstra_shortest_paths(g, src, weight_map(get(&EdgeType::distance, g)).distance_map(
                                make_iterator_property_map(d.begin(), get(vertex_index, g))).visitor(vis));
            }
            catch (int exception) {
                (*ddists)[i+offset][j]=d[dst];
            }
            j++;
        }
        i++;
    }
}

class my_visitor : default_bfs_visitor {
protected:
    set<Vertex> destSet;
public:
    my_visitor(set<Vertex> set)
            : destSet(set) {};

    void initialize_vertex(const Vertex &s, const Graph_t &g) const {}

    void discover_vertex(const Vertex &s, const Graph_t &g) const {}

    void examine_vertex(const Vertex &s, const Graph_t &g) const {}

    void examine_edge(const Edge &e, const Graph_t &g) const {}

    void edge_relaxed(const Edge &e, const Graph_t &g) const {}

    void edge_not_relaxed(const Edge &e, const Graph_t &g) const {}

    void finish_vertex(const Vertex &s, const Graph_t &g)  {
        destSet.erase(s);
        if (destSet.empty())
            throw (2);
    }
};


Perf SSMD_search(vector<vector<double>> *ddists, int offset, int QUANTA, vector<Vertex> &ssources,
                 vector<Vertex> &dests, Graph_t &g, DistanceCache &distanceCache, GraphNodeFactory &graphNodeFact) {
    vector<Vertex> sources(ssources.begin() + offset, ssources.begin() + offset + QUANTA);
    Perf perf;
    perf.sourceSize = sources.size();
    perf.destSize = dests.size();
    int i=0,j=0;
    std::vector<Vertex> p(num_vertices(g));
    std::vector<double> d(num_vertices(g));
    set<Vertex> destsSet;
    for( Vertex d:dests)
        destsSet.insert(d);
    for (Vertex src: sources) {
        my_visitor vis(destsSet);
        cout<<destsSet.size()<<endl;
        fill(p.begin(),p.end(),0);
        fill(d.begin(),d.end(),std::numeric_limits<double>::infinity());
        try{
            cout<<src<<endl;
//            dijkstra_shortest_paths(g, src, weight_map(get(&EdgeType::distance, g)).distance_map(
//                    make_iterator_property_map(d.begin(), get(vertex_index, g))).visitor(vis));
            dijkstra_shortest_paths(g, src, weight_map(get(&EdgeType::distance, g)).distance_map(
                    make_iterator_property_map(d.begin(), get(vertex_index, g))).visitor(vis));
        }
        catch (int exception) {
            j=0;
            for (Vertex dst:dests){
                (*ddists)[i+offset][j]=d[dst];
                j++;
            }
        }
        i++;
    }
}



Perf MSMD_search(vector<vector<double>> *ddists, int offset, int QUANTA, vector<Vertex> &ssources,
                 vector<Vertex> &dests, Graph_t &g, DistanceCache &distanceCache, GraphNodeFactory &graphNodeFact){
    // minimum cost upto
    // goal state from starting
    // state

    // insert the starting index
    // map to store visited node
    vector<Vertex> sources(ssources.begin()+offset,ssources.begin()+offset+QUANTA);
    Perf perf;
    perf.sourceSize=sources.size();
    perf.destSize=dests.size();
    long hit=0, checked=0;
    boost::unordered_set<long> sourceVisited;
    boost::unordered_set<uint> sourceFinished;
    vector<int> srcCount(sources.size(),0);

    //GraphNodeFactory graphNodeFact;
    GraphNodeArray graphNodeArray(num_vertices(g),sources, distanceCache);
    priority_queue<GraphNode*, vector<GraphNode*>, myComp > *costQueue, *bakCostQueue, *swap;
    costQueue=new priority_queue<GraphNode*, vector<GraphNode*>, myComp >();
    boost::unordered_set<Vertex> seen;

    int counter=0;
    int numSettled=0;
    int destsCount=0;
    bool finish=false;
    set<Vertex> ddests(dests.begin(), dests.end());
    Vertex v;
    GraphNode *graphNode;
    for(int i=0;i<sources.size();i++){
        v=sources[i];
        graphNode = graphNodeFact.get(v,i,0.0);
        costQueue->push(graphNode);
        hit ++;
        numSettled ++;
    }
    long maxQueue=0;
    double currDist, toDist, newDist, oldDist;
    Vertex neighbor;
    for(Vertex t:dests){
        if (ddests.count(t)>0){// t is a destination that has not been yet reached for all source
            bool targetFinished=false;
            GraphNode *previous=nullptr;
            // while the queue is not empty
            while (!costQueue->empty()){
                maxQueue=max(maxQueue, (long) costQueue->size());
                counter++;
                // get the top element of the
                // priority heap
                GraphNode *p=costQueue->top();
                costQueue->pop();
                if (sourceFinished.count(p->source)){
                    //check if all destination relative to the source are reached
                    //if yes we do not need to process nodes relative to this source further
                    graphNodeFact.yield(p);
                    continue;
                }
                //settle poped values
                if (!graphNodeArray.isSettled(p->v, p->source)){
                    p->settled=true;
                    graphNodeArray.checkSet(p->v,p->source,p->dist);
                    graphNodeArray.settle(p->v, p->source);
                    //check if the node is one of the destination
                    //if yes we should increase the number of reached
                    if (ddests.count(p->v)>0) {
                        srcCount[p->source]++;
                        if (srcCount[p->source]>= dests.size()) {
                            sourceFinished.insert(p->source);
                            sourceVisited.insert(((long)p->v<<32)+p->source);
                        }
                    }
//                    distanceCache1.insert(sources[p->source], p->v, abs(p->dist));
                    numSettled++;
                }
                //check if all sources know the shortest distance to p->v
                if (graphNodeArray.isFinished(p->v)) {
                    //remove p->v of destination if p->v is a destination
                    ddests.erase(p->v);
                    if (ddests.empty()) {//all dests are attained
                        int j=0;
                        for (Vertex d: dests) {
                            auto range = equal_range(dests.begin(), dests.end(), d);
                            for (int i = 0; i < sources.size(); i++) {
                                (*ddists)[i+offset][j] = abs(graphNodeArray.get(d, i));
                            }
                            ++j;
                        }
                        perf.maxQueue=maxQueue;
                        perf.counter=counter;
                        perf.cacheHitRate = hit*1.0/checked;
                        graphNodeFact.yield(p);
                        while(!costQueue->empty()){
                            GraphNode *pp=costQueue->top();
                            graphNodeFact.yield(pp);
                            costQueue->pop();
                        }
                        delete costQueue;
                        return perf;
                    }
                }
                // pop the element
                // check if the element is part of
                // the goal list
                // if all goals are reached
                // check for the non visited nodes
                // which are adjacent to present node
                if (sourceVisited.count(((long)p->v<<32)+p->source) == 0) {
                    //p->v have not been visited;
                    int deg= out_degree(p->v,g);
                    for (auto ve = boost::out_edges(p->v, g); ve.first != ve.second; ++ve.first) {
                        neighbor = target(*ve.first, g);
                        currDist = graphNodeArray.get(neighbor, p->source);
                        bool updated;
                        toDist = abs(p->dist);
                        newDist = toDist + g[*ve.first].distance;
                        oldDist = graphNodeArray.get(neighbor, p->source);
                        if (isinf(oldDist)) {
                            checked++;
//                           double dist = distanceCache1.find(sources[p->source], neighbor);
                            double dist=-1.0;
                            if (dist > -1.0) {
                                newDist = dist;
                                updated = graphNodeArray.checkSet(neighbor, p->source, dist);
                                graphNodeArray.settle(neighbor, p->source);
                                hit++;
                            } else {
                                updated = graphNodeArray.checkSet(neighbor, p->source, newDist);
                            }
                        } else {
                            updated = graphNodeArray.checkSet(neighbor, p->source, newDist);
                        }
                        if (updated)
                            costQueue->push(graphNodeFact.get(neighbor, p->source,graphNodeArray.get(neighbor,
                                                                                                     p->source)));
                    }
                    sourceVisited.insert(((long)p->v<<32)+p->source);
                }
                graphNodeFact.yield(p);
            }
            if (ddests.empty()) {//all dests are attained
                int j = 0;
                for (Vertex d: dests) {
                    auto range = equal_range(dests.begin(), dests.end(), d);
                    for (int i = 0; i < sources.size(); i++) {
                        (*ddists)[i+offset][j] = graphNodeArray.get(d, i);
                    }
                    ++j;
                }
                perf.maxQueue=maxQueue;
                perf.counter=counter;
                perf.cacheHitRate = hit*1.0/checked;
                while(!costQueue->empty()) {
                    GraphNode *p = costQueue->top();
                    graphNodeFact.yield(p);
                    costQueue->pop();
                }
                delete costQueue;
                return perf;
            }
        }
    }
    if (!costQueue->empty())
        cout<<"BUG"<<endl;
    delete costQueue;
    return perf;
}


enum TaskType {FullProcess, PartialProcess, PartialOT,SuperProcess};
enum AlgoType{SSSD, SSMD, MSMD};

class Task{
public:
    TaskType type;
    Vertex v;
    vector<Vertex> sources;
    vector<Vertex> dests;
    set<Edge> edges;
    float weight;
    Task(TaskType type, Vertex v, vector<Vertex> &sources, vector<Vertex> &dests, set<Edge> &edges, float weight):type(type), v(v), sources(sources),
                                                                                                                  dests(dests), edges(edges), weight(weight){}
    ~Task(){
        sources.clear();
        dests.clear();
        edges.clear();
    }
};

struct TaskPaircomp {
    // Comparator function
    int operator()(Task *l, Task *r) const {
        if (l->v == r->v)
            return 0;
        if (l->weight < r->weight) {
            return 1;
        }
        return -1;
    }
};

class FullTask: public Task{
public:
    FullTask(Vertex v, vector<Vertex> &sources, vector<Vertex> &dests, set<Edge> &edges):Task(FullProcess,v,sources, dests, edges, (float)sources.size())
    {};
};

class PartialTask: public Task{
public:
    std::atomic<int> *sharedCnt;
    vector<vector<double>> *dists;
    int index;
    int offset;
    int QUANTA;
    PartialTask(std::atomic<int> *shared, vector<vector<double>> *dists, int offset, int QUANTA, Vertex v,
                vector<Vertex> sources, vector<Vertex> dests, set<Edge> edges, int index):
            Task(PartialProcess,v,sources, dests,edges,(float)sources.size()), sharedCnt(shared),
            dists(dists), index(index), offset(offset), QUANTA(QUANTA){this->type=PartialProcess;}
};

class PartialCurvature: public Task{
public:
    std::atomic<int> *sharedCnt;
    vector<vector<double>> *dists;
    int index;
    int offset;
    int QUANTA;
    PartialCurvature(std::atomic<int> *shared, vector<vector<double>> *dists, int offset, int QUANTA, Vertex v,
                     vector<Vertex> sources, vector<Vertex> dests, set<Edge> edges, int index):
            Task(PartialOT,v,sources, dests,edges, (float)sources.size()-0.1), sharedCnt(shared), dists(dists),
            index(index), offset(offset), QUANTA(QUANTA){this->type=PartialOT;}

};
using TaskPriorityQueue = PriorityBlockingCollection<Task *, PriorityContainer<Task*, TaskPaircomp>>;
set<Task *, TaskPaircomp> tasksToDoSet;

void copyTaskQueue(set<Task *, TaskPaircomp> &intaskPriorityQueue, TaskPriorityQueue &outtaskPriorityQueue){
    int i=0;
    for (Task *task: intaskPriorityQueue){
        outtaskPriorityQueue.try_add(task);
        i++;
    }
}

//template <class Graph> using TaskPriorityQueue = PriorityBlockingCollection<Task *, PriorityContainer<Task<Graph>*, TaskPaircomp<Graph>>>;

atomic<long>  numProcessedVertex{0}, numProcessedEdge{0}, numProcessedPath{0};
atomic<float> threshold{0.05};
long numbOfShortPaths=0;

constexpr double EPS=1e-3;

void calcCurvature1(double alpha, Vertex v, vector<vector<double>> &MatDist, set<Edge> &edges, vector<Vertex> &sources,
                   vector<Vertex> &dests, Graph_t &g){

    vector<double> distributionA;
    vector<double> distributionB;
    double wd;
    for (Edge e: edges) {
        Vertex src = source(e, g);
        Vertex neighbor = target(e, g);
        pair<AdjacencyIterator, AdjacencyIterator> nB = adjacent_vertices(neighbor, g);
        set<Vertex> localDests;
        localDests.insert(neighbor);
        for (; nB.first != nB.second; localDests.insert(*nB.first++));
        vector<vector<double>> vertDist(MatDist.size(), vector<double>(localDests.size(),
                                                                       std::numeric_limits<double>::infinity()));

        // Remplissage de la matrice locale des plus court chemin
        int cntD = 0, destIndex = 0;
        for (Vertex dest: localDests) {
            auto it=find(dests.begin(),dests.end(),dest);
            if (it==dests.end()){
                cout<<dest<<" ,PROBLEMM!!!"<<endl;
                for(Vertex d:dests)
                    cout<<d;
                cout<<endl;
            }
            int indexj = std::distance(it,dests.begin());
            for (int j = 0; j < MatDist.size(); j++) {
                vertDist[j][cntD] = MatDist[j][indexj];
            }

            if (dest == neighbor)
                destIndex = cntD;
            cntD++;
        }

//
/*
        int sA = vertDist.size(), sB = vertDist[0].size();

        double uA = (double) (1 - alpha) / (sA - 1);
        double uB = (double) (1 - alpha) / (sB - 1);
        distributionA.resize(sA);
        fill_n(distributionA.begin(), sA, uA);
        distributionB.resize(sB);
        fill_n(distributionB.begin(), sB, uB);
        auto range = equal_range(sources.begin(), sources.end(), src);
        int index = range.first - sources.begin();

        distributionA[index] = alpha;
        distributionB[destIndex] = alpha;

*/
//

///////////////////////////////////////////////

        // X distribution vector
        int sA = vertDist.size();
        distributionA.resize(sA);

        pair<AdjacencyIterator, AdjacencyIterator> nA = adjacent_vertices(src, g);
        set<Vertex> Sources;
        Sources.insert(src);
        for (; nA.first != nA.second; Sources.insert(*nA.first++));
        double sws = 0;
        int cnt=0;
        for (Vertex s: Sources) {
            if (src == s){
                distributionA[cnt] = 0;
            }else {
                auto e = edge(src, s, g);
 //               cout << "distance= "<< g[e.first].distance << endl;
                distributionA[cnt] = g[e.first].distance;
                sws = sws + g[e.first].distance;
            }
            cnt++;
        }

        for(int i = 0; i<distributionA.size(); i++) {
            distributionA[i] = distributionA[i]/sws;
        }
        // Y destribution vertor
        int sB = vertDist[0].size();
        distributionB.resize(sB);

        double swd = 0;
        cnt=0;
        for (Vertex dest: localDests) {
            if (dest == neighbor){
                distributionB[cnt] = 0;
            }else {
                auto ed = edge(neighbor, dest, g);
                distributionB[cnt] = g[ed.first].distance;
                swd = swd + g[ed.first].distance;
            }
            cnt++;
        }
        for(int i = 0; i<distributionB.size(); i++) {
            distributionB[i] = distributionB[i]/swd;
        }
        wd = compute_EMD(distributionA, distributionB, vertDist);
        g[e].oot =g[e].ot;
        g[e].ocurv =g[e].curv;
        if (isnan(wd)) {
            g[e].ot = 9999.0;
            g[e].curv = 9999.0;
            cout<<"PROBLEM"<<endl;
        } else {
            float tmpDist=g[e].distance;
            if (tmpDist < EPS) {
                g[e].curv = g[graph_bundle].avgCurv;
            } else {
                g[e].curv = 1 - wd / g[e].distance;
            }
        }
        if (isinf(g[e].curv))
            cout<<"BUG"<<endl;

        distributionB.clear();
        distributionA.clear();

        // } else {
        //     g[e].curv = 0.0;
        //     g[e].ot = 0.0;
        //     g[e].distance = EPS / 2;
        // }
        numProcessedEdge++;
    }
}

void calcCurvature(double alpha, Vertex v, vector<vector<double>> &MatDist, set<Edge> &edges, vector<Vertex> &sources,
                   vector<Vertex> &dests, Graph_t &g){

    double wd;
    for (Edge e: edges) {
        Vertex src = source(e, g);
        Vertex neighbor = target(e, g);

        pair<AdjacencyIterator, AdjacencyIterator> nB = adjacent_vertices(neighbor, g);
        set<Vertex> localDests;
        localDests.insert(neighbor);
        for (; nB.first != nB.second; localDests.insert(*nB.first++));
        vector<vector<double>> vertDist(MatDist.size(), vector<double>(localDests.size(),
                                                                       std::numeric_limits<double>::infinity()));
        vector<vector<double>> optTransport(MatDist.size(), vector<double>(localDests.size(),
                                                                       std::numeric_limits<double>::infinity()));

        int cntD = 0, destIndex = 0;
        for (Vertex dest: localDests) {
            auto it=find(dests.begin(), dests.end(), dest);
            int indexj = it - dests.begin();
            for (int j = 0; j < MatDist.size(); j++) {
                vertDist[j][cntD] = MatDist[j][indexj];
            }
            if (dest == neighbor)
                destIndex = cntD;
            cntD++;
        }

        int sA = vertDist.size(), sB = vertDist[0].size();
        //if (sB > 1) {

        double uA = (double) (1 - alpha) / (sA - 1);
        double uB = (double) (1 - alpha) / (sB - 1);
        vector<double> distributionA(sA,uA);
        vector<double> distributionB(sB,uB);
        auto wrapIter=find(sources.begin(), sources.end(), src);
        long index;
        if (wrapIter != sources.end()){
            index= wrapIter - sources.begin();
            distributionA[index] = alpha;
        } else {
            cout<<"BUG"<<endl;
        }
        distributionB[destIndex] = alpha;
        double cost;

        wd = compute_EMD(distributionA, distributionB, vertDist);
        g[e].oot =g[e].ot;
        g[e].ocurv =g[e].curv;
        if (isnan(wd)) {
            g[e].ot = 9999.0;
            g[e].curv = 9999.0;
            cout<<"PROBLEM"<<endl;
        } else {
            g[e].ot = wd;
//            float tmpDist=g[e].distance;
            g[e].curv = (1 - wd / g[e].distance)/(1-alpha);
        }
//        cout<<wd<<","<<(1 - wd / g[e].distance)<<","<<g[e].curv<<endl;
//        uA = (double) 1 / (sA - 1);
//        uB = (double) 1 / (sB - 1);
//        fill_n(distributionA.begin(), sA, uA);
//        fill_n(distributionB.begin(), sB, uB);
//        distributionA[index] = 0.0;
//        distributionB[destIndex] = 0.0;
//        wd = compute_EMD(distributionA, distributionB, vertDist);
//        cout <<(1-wd/g[e].distance)<<","<<g[e].curv<<endl;
        distributionB.clear();
        distributionA.clear();

        // } else {
        //     g[e].curv = 0.0;
        //     g[e].ot = 0.0;
        //     g[e].distance = EPS / 2;
        // }
        numProcessedEdge++;
    }
}

auto t1=high_resolution_clock::now();
void process(AlgoType algo, int  threadIndex, Graph_t *g, DistanceCache *distanceCache, TaskPriorityQueue *tasksToDo,
              atomic<int> *runningTaskCount) {
    GraphNodeFactory graphNodeFact;
    Task *task;
    long totalTime1 = 0, totalTime2 = 0;
    int QUANTA=120;
    long numberProcessedPath=0;
    std::chrono::nanoseconds elapsed=t1-t1;
    std::chrono::nanoseconds elapsed2=t1-t1;
    auto t2=t1;
    auto t3=t1;
    do{
        while (tasksToDo->try_take(task,std::chrono::milliseconds(1000)) == BlockingCollectionStatus::Ok) {
            //        tasksToDo.try_take(task);
            numberProcessedPath=0;
            Vertex s = task->v;
            Perf perf1;// perf2;
            bool done = false;
            vector<Vertex> sources(task->sources), dests(task->dests);
            set<Edge> edges(task->edges);
            vector<vector<double>> *MatDist1;// *MatDist2;
            //        vector<vector<double>> *MatDist;

            int ssize = sources.size(), dsize = dests.size(), allsize=ssize*dsize;
            switch (task->type) {
                case FullProcess: {
                    auto *fullTask = (FullTask *) task;
                    (*runningTaskCount)++;
                    MatDist1 = new vector<vector<double>>(ssize, vector<double>(dsize,
                                                                                std::numeric_limits<double>::infinity()));
//                    MatDist2= new vector<vector<double>>(ssize, vector<double>(dsize,std::numeric_limits<double>::infinity()));
                    if (ssize < QUANTA) {
                        switch(algo){
                            case SSSD:
                                perf1 = SSSD_search(MatDist1, 0, sources.size(), sources, dests, *g, *distanceCache,
                                                graphNodeFact);
                                break;
                            case SSMD:
                                perf1 = SSMD_search(MatDist1, 0, sources.size(), sources, dests, *g, *distanceCache,
                                                    graphNodeFact);
                                break;
                            case MSMD:
                                perf1 = MSMD_search(MatDist1, 0, sources.size(), sources, dests, *g, *distanceCache,
                                            graphNodeFact);
                                break;
                            default:
                                perf1 = MSMD_search(MatDist1, 0, sources.size(), sources, dests, *g, *distanceCache,
                                                    graphNodeFact);
                        }
                        auto *sharedCnt = new std::atomic<int>();
                        *sharedCnt = 0;
                        auto *ptask = new PartialCurvature(sharedCnt, MatDist1, 0, edges.size(), fullTask->v, sources, dests, edges, 0);
                        tasksToDo->try_add((Task *) ptask);
                        //                    delete MatDist2;
                    } else {
                        auto  *sharedCnt = new std::atomic<int>();
                        *sharedCnt =0;
                        int iterNum = sources.size() / QUANTA;
                        auto begin = sources.begin();
                        for (int i = 0; i < sources.size(); i = i + QUANTA) {
                            auto *ptask = new PartialTask(sharedCnt, MatDist1, i,std::min(QUANTA, (int)sources.size()-i),s,
                                                                 sources, dests, edges,*sharedCnt);
                            tasksToDo->try_add((Task *) ptask);
                            (*sharedCnt)++;
                        }
                    }
                    delete fullTask;
                    break;
                }
                case PartialProcess:{
                    auto *partialTask = (PartialTask *) task;
                    //                t2 = high_resolution_clock::now();
//                cout<<"Partial Task:"<<partialTask->v<<":"<<*(partialTask->sharedCnt)<<", thread#:"<<threadIndex<<endl;
//                    perf1 = MSMD_search(partialTask->dists, partialTask->offset, partialTask->QUANTA, sources, dests,
//                                        *g, *distanceCache, graphNodeFact);
                    switch(algo){
                        case SSSD:
                            perf1 = SSSD_search(partialTask->dists, partialTask->offset, partialTask->QUANTA, sources, dests,
                                                *g, *distanceCache, graphNodeFact);
                            break;
                        case SSMD:
                            perf1 = SSMD_search(partialTask->dists, partialTask->offset, partialTask->QUANTA, sources, dests,
                                                *g, *distanceCache, graphNodeFact);
                            break;
                        case MSMD:
                            perf1 = MSMD_search(partialTask->dists, partialTask->offset, partialTask->QUANTA, sources, dests,
                                                *g, *distanceCache, graphNodeFact);
                            break;
                        default:
                            perf1 = MSMD_search(MatDist1, 0, sources.size(), sources, dests, *g, *distanceCache,
                                                graphNodeFact);
                    }

                    //                t3 = high_resolution_clock::now();
                    done = true;
                    (*(partialTask->sharedCnt))--;
                    if (*(partialTask->sharedCnt) == 0) {
                        delete partialTask->sharedCnt;
                        auto *sharedCnt = new std::atomic<int>();
                        *sharedCnt = 0;
                        int iterNum = edges.size() / QUANTA;
                        auto begin = edges.begin();
                        int i=0,j=0;
                        set<Edge> partialEdges;
                        for (Edge e:edges){
                            if (i==QUANTA){
                                auto *ptask = new PartialCurvature(sharedCnt, partialTask->dists,j ,QUANTA, s, sources, dests,
                                                                               partialEdges,*sharedCnt);
                                tasksToDo->try_add((Task *) ptask);
                                (*sharedCnt)++;
                                partialEdges.clear();
                                i=0;
                                j+=QUANTA;
                            }
                            partialEdges.insert(e);
                            i++;
                        }
                        if (!partialEdges.empty()){
                            auto *ptask = new PartialCurvature(sharedCnt, partialTask->dists, j, partialEdges.size(), s, sources, dests,
                                                                           partialEdges,*sharedCnt);
                            tasksToDo->try_add((Task *) ptask);
                            partialEdges.clear();
                        }
                    }
                    delete partialTask;
                    break;
                }
                case PartialOT:{
                    auto *partialOT = (PartialCurvature *) task;
                    t2 = high_resolution_clock::now();
                    calcCurvature(ALPHA,partialOT->v, *(partialOT->dists),partialOT->edges, sources,dests,*g);
                    t3= high_resolution_clock::now();
                    done = true;
                    if (*(partialOT->sharedCnt) == 0) {
                        numberProcessedPath=sources.size()*dests.size();
                        numProcessedPath+=numberProcessedPath;
                        numProcessedVertex++;
                        numProcessedEdge +=partialOT->edges.size()+partialOT->offset;
                        done = true;
                        delete partialOT->dists;
                        delete partialOT->sharedCnt;
                        (*runningTaskCount)--;
                        elapsed +=(t2 - t1);
                        elapsed2+= (t3 - t2);

//                        cout<<"#Vert:"<<numProcessedVertex<<" #Edg:"<<numProcessedEdge<<" %Pth:"<<numProcessedPath*1.0/numbOfShortPaths*100<<"% time:"<<duration_cast<microseconds>(
//                                elapsed).count()<<endl;
                        if (threshold * numbOfShortPaths < numProcessedPath){
                            threshold= threshold +0.05;
                            cout<<"Time passsed to execute :"<< numProcessedPath*1.0/  numbOfShortPaths<<"% is "<<duration_cast<microseconds>(
                                    elapsed).count()<<" and "<<duration_cast<microseconds>(elapsed2).count()<<","<<duration_cast<microseconds>(
                                    elapsed).count()*1.0/(duration_cast<microseconds>(elapsed2).count()+duration_cast<microseconds>(elapsed).count())<<endl;
                        }
                    } else {
                        (*(partialOT->sharedCnt))--;
                    }
                    delete partialOT;
                    break;
                }
                case SuperProcess:
                    break;
            }
        }
    } while(*runningTaskCount!=0);
}

struct vertexpaircomp {
    // Comparator function
    bool operator()(const pair<Vertex, int>& l, const pair<Vertex, int>& r) const
    {
        if (l.second != r.second) {
            return l.second < r.second;
        }
        return true;
    }
};


long generateTasks(Graph_t &g, TaskPriorityQueue &tasksToDo){
    set<long> edgeSet;
    int removed=0, added=0;
    AdjacencyIterator nfirst, nend;
    VertexIterator v,vend;
    set<pair<Vertex, int>, vertexpaircomp> degreeSorted;
    int numbOfEdges=0;
    for (tie(v, vend) = vertices(g); v != vend; v++) {
        degreeSorted.insert(make_pair(*v, -degree(*v, g)));
    }
    set<Vertex> sourcesSet, destsSet;
    set<Task *, TaskPaircomp> taskSet;
    set<Edge> edgesToProcess;
    for (auto p: degreeSorted) {
        Vertex src = p.first;
        sourcesSet.insert(src);
        for (auto ve = boost::out_edges(src, g); ve.first != ve.second; ++ve.first) {
            bool found = true;
            Vertex dest = target(*ve.first, g);
            if (src!=dest){
                destsSet.insert(dest);
                sourcesSet.insert(dest);
                long key = (src << 32) + dest;
                if (edgeSet.insert(key).second) {
                    found = false;
                }
                key = (dest << 32) + src;
                if (edgeSet.insert(key).second) {
                    found = false;
                }
                if (!found) {
                    // the edge have not been processed
                    tie(nfirst, nend) = adjacent_vertices(dest, g);
                    destsSet.insert(nfirst, nend);
                    edgesToProcess.insert(*ve.first);
                    added++;
                } else {
                    removed++;
                }
            }
        }
        vector<Vertex> sources(sourcesSet.begin(), sourcesSet.end()), dests(destsSet.begin(), destsSet.end());
        if (!edgesToProcess.empty()) {
            numbOfEdges +=edgesToProcess.size();
            numbOfShortPaths +=sources.size()*destsSet.size();
            auto task = new FullTask(src, sources, dests, edgesToProcess);
            tasksToDo.try_add((Task *) task);
        }
        sourcesSet.clear();
        destsSet.clear();
        edgesToProcess.clear();
    }
    if (edgeSet.size()!= 2*num_edges(g))
        cout<<"Edge Number Mismatch !"<<endl;
    cout<<"Number of tasks:"<<tasksToDo.size()<<" ,numVertices="<<num_vertices(g)<<" ,numbOfEdges="<<numbOfEdges<<", numbOfShortPaths="<<numbOfShortPaths<<endl;
    return numbOfShortPaths;
}

long generateTasks1(Graph_t &g, TaskPriorityQueue &tasksToDo){
    set<long> edgeSet;
    int removed=0, added=0;
    AdjacencyIterator nfirst, nend;
    VertexIterator v,vend;
    set<pair<Vertex, int>, vertexpaircomp> degreeSorted;
    int numbOfEdges=0;
    for(auto ve = edges(g);ve.first!=ve.second;++ve.first){
        vector<Vertex> sources,dests;
        set<Edge> edgesToProcess;
        Vertex src=source(*(ve.first), g);
        Vertex dst=target(*(ve.first),g);
        sources.push_back(src);
        for(auto e= adjacent_vertices(src,g);e.first!=e.second;++e.first){
            sources.push_back(*(e.first));
        }
        dests.push_back(dst);
        for(auto e= adjacent_vertices(dst,g);e.first!=e.second;++e.first){
            dests.push_back(*(e.first));
        }
        edgesToProcess.insert(*(ve.first));
        numbOfEdges +=edgesToProcess.size();
        numbOfShortPaths +=sources.size()*dests.size();
        auto task = new FullTask(src, sources, dests, edgesToProcess);
        tasksToDo.try_add((Task *) task);

    }
    cout<<"Number of tasks:"<<tasksToDo.size()<<" ,numVertices="<<num_vertices(g)<<" ,numbOfEdges="<<numbOfEdges<<", numbOfShortPaths="<<numbOfShortPaths<<endl;
    return numbOfShortPaths;
}

struct Predicate {// both edge and vertex
    typedef typename graph_traits<Graph_t>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph_t>::edge_descriptor Edge;

    explicit Predicate(Graph_t *g): g(g){};
    Predicate()= default;
    bool operator()(Edge e) const      {return (*g)[e].active;}
    bool operator()(Vertex vd) const { return (*g)[vd].active; }
    Graph_t *g;
};

using Filtered_Graph_t = boost::filtered_graph<Graph_t, Predicate, Predicate>;

void k_core2(Graph_t &gin, Graph_t &gout, unsigned int k){
    VertexIterator v,vend;
    EdgeIterator e,eend;
    auto *pgraph=new Graph_t();
    vector<unsigned long> degrees(num_vertices(gin));
    map<Edge,bool> allEdges;
    set<pair<Vertex, int>, vertexpaircomp> degreeSorted;
    copy_graph(gin,*pgraph);
    bool proceed;
    //find self-loops and remove them.
    for(auto ve = boost::edges(*pgraph);ve.first!=ve.second;++ve.first){
        Vertex src=source(*(ve.first),*pgraph);
        Vertex dst=target(*(ve.first),*pgraph);
        if (src==dst)
            (*pgraph)[*ve.first].active=false;
    }
    do{
        proceed=false;
        degreeSorted.clear();
        for(tie(v,vend)= vertices(*pgraph);v!=vend;v++) {
            degreeSorted.insert(make_pair(*v, degree(*v, *pgraph)));
        }
        for(auto p:degreeSorted){
            if (p.second<k){
                proceed=true;
                (*pgraph)[p.first].active=false;
                for (auto ve = boost::out_edges(p.first, *pgraph); ve.first != ve.second; ++ve.first) {
                    (*pgraph)[*ve.first].active=false;
                }
            } else
                break;
        }
        Predicate predicate(pgraph);
        Filtered_Graph_t fg(*pgraph, predicate, predicate);
        Graph_t *toDelete=pgraph;
        pgraph=new Graph_t();
        copy_graph(fg,*pgraph);
        toDelete->clear();
        delete toDelete;
    } while(proceed);
    copy_graph(*pgraph,gout);
    for(auto ve = boost::edges(gout);ve.first!=ve.second;++ve.first){
        Vertex src=source(*(ve.first),*pgraph);
        Vertex dst=target(*(ve.first),*pgraph);
        if (src==dst)
            gout[*ve.first].active=false;
    }

    cout<<"num vertices: "<<num_vertices(gout)<<", num edges: "<<num_edges(gout)<<endl;
    delete pgraph;
}


void calcStats(Graph_t &g, int componentNum, vector<int> &components) {
    double mindist = 999999.0;
    double maxdist = -1.0;
    vector<double> sumCurvperComponent(componentNum, 0.0), sum2CurvperComponent(componentNum, 0.0);
    vector<int> componentSize(componentNum, 0);
    double sumCurv = 0.0, sum2Curv = 0.0;
    double sumDist =0.0, sum2Dist = 0.0;
    Edge maxEdge, minEdge;
    auto es = edges(g);
    for (auto eit = es.first; eit != es.second; ++eit) {
        Vertex src = source(*eit, g), dst = target(*eit, g);
        float tmpDist=g[*eit].distance;
        if (tmpDist > maxdist) {
            maxdist = tmpDist;
            maxEdge = *eit;
        }
        sumDist += tmpDist;
        sum2Dist += tmpDist * tmpDist;
        if (tmpDist < mindist) {
            mindist = tmpDist;
        }
        float tmpCurv=g[*eit].curv;
        if (isinf(tmpCurv)){
            cout<<"inf edge:"<<source(*eit,g)<<":"<<target(*eit,g)<<endl;
        }
        sumCurv += tmpCurv;
        sum2Curv += tmpCurv * tmpCurv;
        int comp = components[src];
        sumCurvperComponent[comp] = sumCurvperComponent[comp] + g[*eit].curv;
        componentSize[comp]++;
        sum2CurvperComponent[comp] = sum2CurvperComponent[comp] + g[*eit].curv * g[*eit].curv;
    }
    for (int j = 0; j < componentNum; j++) {
        if (componentSize[j] > 0) {
            cout << "Size:" << componentSize[j] << " ,avgCurv[" << j << "]="
                 << sumCurvperComponent[j] * 1.0 / componentSize[j] << ", stdCurv[" << j << "]=" <<
                 sqrt(sum2CurvperComponent[j] * 1.0 / componentSize[j] -
                      (sumCurvperComponent[j] / componentSize[j]) * (sumCurvperComponent[j] / componentSize[j]))
                 << endl;
        }
    }
}

bool updateDistances(Graph_t &g, double &oldrescaling) {
    auto es = edges(g);
    double delta = 0.1;
    double sumWeights = 0.0;
    int numEdgesUnfiltered = 0;
    bool surgery = false;
    float maxdist=0.0, mindist=99999.9, sumDist=0.0, sumCurv=0.0, sum2Dist=0.0, sum2Curv=0.0;
    Edge maxEdge;
    for (auto eit = es.first; eit != es.second; ++eit) {
        g[*eit].odistance=g[*eit].distance;
        g[*eit].distance = g[*eit].distance * (1 - g[*eit].curv/2);
        if (g[*eit].distance > maxdist) {
            maxdist =g[*eit].distance;
            maxEdge = *eit;
        }
        if (g[*eit].distance < mindist) {
            mindist = g[*eit].distance;
        }
        sumDist += g[*eit].distance;
        sum2Dist += g[*eit].distance * g[*eit].distance;
        sumCurv +=g[*eit].curv;
        sum2Curv +=g[*eit].curv*g[*eit].curv;
    }
    g[graph_bundle].avgDist = sumDist*1.0/ num_edges(g);
    g[graph_bundle].stdDist = sqrt(sum2Dist*1.0/ num_edges(g) - g[graph_bundle].avgDist * g[graph_bundle].avgDist);
    g[graph_bundle].rescaling= 1.0/g[graph_bundle].avgDist;
    g[graph_bundle].rstdDist= g[graph_bundle].stdDist*g[graph_bundle].rescaling;
    g[graph_bundle].avgCurv = sumCurv*1.0/ num_edges(g) ;
    g[graph_bundle].stdCurv = sqrt(sum2Curv*1.0 / num_edges(g) - g[graph_bundle].avgCurv * g[graph_bundle].avgCurv);

    cout << "maxdist:" << source(maxEdge,g)<<":"<<target(maxEdge,g)<<"="<<maxdist<<endl;
    cout << "avgCurv=" << g[graph_bundle].avgCurv << ", stdCurv=" << g[graph_bundle].stdCurv << endl;
    cout << "avgDist=" << g[graph_bundle].avgDist << ", stdDist=" << g[graph_bundle].stdDist << endl;
    cout << "rescaling=" << g[graph_bundle].rescaling<<", rstdDist="<<g[graph_bundle].rstdDist<<endl;
    //Now rescaling
    for (auto eit = es.first; eit != es.second; ++eit) {
        g[*eit].distance= g[*eit].distance*g[graph_bundle].rescaling;
        //Check surgery type 1
/*        if ((g[*eit].distance <= EPS) && (!g[*eit].surgery)) { //we need a surgery of type 1
            Vertex src = source(*eit, g), dst = target(*eit, g);
            surgery = false;
            g[*eit].surgery = true;
            g[*eit].distance = EPS;
            cout << "Surgery Type 1: " << src << ":" << dst << ", Curvature:" << g[*eit].curv<<endl;
            cout<<g[src].name<<":"<<g[dst].name<<endl;
        }
        if ((g[*eit].distance >= 1+3*g[graph_bundle].rstdDist) && (!g[*eit].surgery)){
            Vertex src = source(*eit, g), dst = target(*eit, g);
            cout << "Surgery Type 2: " << src << ":" << dst << ", Curvature:" << g[*eit].curv <<
                 ", Dist;"<< g[*eit].distance<<endl;
            //           surgery=true;
//            g[*eit].surgery=true;
//            g[*eit].active=false;
//            cout<<g[src].country<<":"<<g[dst].country<<","<<g[src].asnumber<<":"<<g[dst].asnumber<<endl;
            cout<<g[src].name<<":"<<g[dst].name<<endl;

        }*/
    }
    oldrescaling = g[graph_bundle].rescaling;
    return surgery;
}


bool triangle_checker(Graph_t &g,int type){
    map<Vertex, set<Vertex>> A;
    bool status=true;
    int numTriangle=0;
    VertexIterator  v, vend;
    for (tie(v, vend) = vertices(g); v != vend; v++){
        A[*v].empty();
    }
    AdjacencyIterator vAdj, vAdjEnd;
    set<Vertex> intersect;
    for (tie(v, vend) = vertices(g); v != vend; v++){
        for (tie(vAdj, vAdjEnd) = adjacent_vertices(*v, g); vAdj != vAdjEnd; vAdj++) {
            intersect.clear();
            if (*v < *vAdj){
                set_intersection(A[*v].begin(), A[*v].end(), A[*vAdj].begin(), A[*vAdj].end(),
                                 inserter(intersect, intersect.begin()));
                for (Vertex vv: intersect) {
                    numTriangle ++;
                    auto pp = edge(*v, vv, g);
                    auto pp1 = edge(vv, *vAdj, g);
                    auto pp2 = edge(*v, *vAdj, g);
                    if ((pp.second) && (pp1.second) && (pp2.second)) {
                        if (type == 0) {
                            if ((g[pp.first].distance + g[pp1.first].distance < g[pp2.first].distance) ||
                                (g[pp1.first].distance + g[pp2.first].distance < g[pp.first].distance) ||
                                (g[pp.first].distance + g[pp2.first].distance < g[pp1.first].distance)) {
                                cout << "Violation" << *v << "," << *vAdj << "," << vv << endl;
                                status=false;
                            }
                        } else {
                            if ((g[pp.first].edist + g[pp1.first].edist < g[pp2.first].edist) ||
                                (g[pp1.first].edist + g[pp2.first].edist < g[pp.first].edist) ||
                                (g[pp.first].edist + g[pp2.first].edist < g[pp1.first].edist)) {
                                cout << "Violation" << *v << "," << *vAdj << "," << vv << endl;
                                status=false;
                            }
                        }
                    }
                }
                A[*vAdj].insert(*v);
            }
        }
    }
    cout<<"Num Triangle:"<<numTriangle<<endl;
    return status;
}


void ricci_flow(Graph_t *g, int numIteration, int iterationIndex, const string& path, AlgoType algo) {
    Graph_t *ginter;
    double oldRescaling = 1.0;
    DistanceCache distanceCache(num_vertices(*g));
    vector<int> component(num_vertices(*g));
    int numComponent = connected_components(*g, &component[0]);
    ofstream logFile;
    ofstream outFile;
    atomic<int> runningTasksCount{0};
    auto tt1 = high_resolution_clock::now(), t2 = tt1;
    long numOfShortestPaths;
//    long numbOfShortPaths=generateTasks(*g, tasksToDoSet);
    for (int index = iterationIndex; index < iterationIndex + numIteration; index++) {
        tt1 = high_resolution_clock::now();
        TaskPriorityQueue tasksToDo;
        numOfShortestPaths = generateTasks(*g, tasksToDo);
//      numOfShortestPaths = generateTasks1(*g, tasksToDo);
        string logFilename = path + "/processed/logFile." + to_string(index + 1) + ".log";
        logFile.open(logFilename.c_str(), ofstream::out);
        string outFilename = path + "/processed/processed." + to_string(index + 1) + ".graphml";
        outFile.open(outFilename.c_str(), ofstream::out);
        cout << "Index:" << index << " ";
        numProcessedVertex = 0;
        numProcessedEdge = 0;
//    uint num_core = std::thread::hardware_concurrency();
        uint        num_core=1;
        int offset = 0;
        int k = 0;
//    num_core=1;
        vector<thread> threads(num_core);
        for (int i = 0; i < num_core; i++) {
            threads[i] = std::thread(process, algo, i, g, &distanceCache, &tasksToDo, &runningTasksCount);
        }
        for (int i = 0; i < num_core; i++) {
            threads[i].join();
        }
//        calcStats(*g, numComponent, component);
        boost::dynamic_properties dpout;
        if (updateDistances(*g, oldRescaling)) {
            ginter = new Graph_t();
            Predicate predicate(g);
            Filtered_Graph_t fg(*g, predicate, predicate);
            copy_graph(fg, *ginter);
            dpout = gettingProperties<Graph_t,VertexType,EdgeType >(*ginter);
            write_graphml(outFile, *ginter, dpout, true);
            (g)->clear();
            copy_graph(*ginter, *g);
//            g->clear();
//            delete g;
//            g = ginter;
        } else {
            dpout = gettingProperties<Graph_t,VertexType,EdgeType>(*g);
            write_graphml(outFile, *g, dpout, true);
        }

//        write_graphml(outFile, *g, dpout, true);
        logFile.close();
        outFile.close();
        t2 = high_resolution_clock::now();
        auto executionTime = duration_cast<microseconds>(t2 - tt1);
        cout << "Execution Time=" << executionTime.count() << ", avg per node="
             << executionTime.count() * 1.0 / num_vertices(*g) << endl;
        cout << "avg per link=" << executionTime.count() * 1.0 / num_edges(*g) << ", avg per path="
             << executionTime.count() * 1.0 / numOfShortestPaths << endl;
    }
}
#endif //CURVATURE_CURVATUREHANDLER_H