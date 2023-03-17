#ifndef CURVATURE_CURVATUREHANDLER_H
#define CURVATURE_CURVATUREHANDLER_H
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphml.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/connected_components.hpp>
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
#define ALPHA 0.0

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
    float *distanceMat;
    int size;
public:
    DistanceCache(int numVertices):size(numVertices*numVertices){
        distanceMat=(float *)calloc(size, sizeof(float));
    }
    ~DistanceCache(){
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
    long maxQueue=0.0;
    long counter = 0.0;
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
    int graphSize;
    vector<Vertex> &sources;
    double *nodeArray;
//    double *minDist;
    int *settledCount;
    uint *minSource;
    DistanceCache &distanceCache;
    GraphNodeArray(int graphSize, vector<Vertex> &sources, DistanceCache &distanceCache): graphSize(graphSize), sources(sources),
                                                                                          distanceCache(distanceCache){
        nodeArray = new (double[graphSize*(sources.size()+1)]);
        for (int i=0;i<graphSize*(sources.size()+1); nodeArray[i++]=std::numeric_limits<double>::infinity());
//        minDist=&(nodeArray[graphSize*sources.size()]);
        minSource= new (unsigned int[graphSize]);
        settledCount=new (int[graphSize]);
        for (int i=0;i<graphSize;settledCount[i++]=0);
    };

    ~GraphNodeArray(){
        delete[] nodeArray;
        delete[] minSource;
        delete[] settledCount;
    }

    double get(uint vind, uint sourceind){
        return nodeArray[sourceind*graphSize+vind];
    };

    bool checkSet(uint vind, uint sourceind, double dist){
        if (nodeArray[sourceind*graphSize+vind]>=dist){
            nodeArray[sourceind*graphSize+vind]=dist;
            return true;
        }
        return false;
    }

    bool isSettled(uint vind, uint sourceind) {
        // Assume that a negative or 0 distance means that it is settled
        if (get(vind, sourceind)>0.0) {
            return false;
        } else {
            return true;
        }
    }

    bool settle(uint vind, uint sourceind) {
        nodeArray[sourceind*graphSize+vind]=-nodeArray[sourceind*graphSize+vind];

        settledCount[vind]++;
//        distanceCache.set(sourceind,vind,);
        return true;
    }

    bool isFinished(uint vind){
        if (settledCount[vind]>=sources.size()){
            return true;
        } else {
            return false;
        }
    }

};

class GraphNodeFactory{
public:
    GraphNodeFactory(){};
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

    long number(){
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


Perf multisource_uniform_cost_search_seq1(vector<vector<double>> *ddists, int offset, int QUANTA, vector<Vertex> &ssources,
                                          vector<Vertex> &dests, Graph_t &g, DistanceCache &distanceCache, GraphNodeFactory &graphNodeFact){
    // minimum cost upto
    // goal state from starting
    // state

    // insert the starting index
    // map to store visited node
    vector<Vertex> sources(ssources.begin()+offset,ssources.begin()+offset+QUANTA);
    set<Vertex> landMarks;
    Perf perf;
    perf.sourceSize=sources.size();
    perf.destSize=dests.size();
    long hit=0, checked=0;
    boost::unordered_set<Vertex> visited;
    boost::unordered_set<Vertex> settledNodes;
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
            GraphNode *previous=NULL;
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
            Task(PartialProcess,v,sources, dests,edges,(float)sources.size()), dists(dists),
            QUANTA(QUANTA), offset(offset), sharedCnt(shared), index(index){this->type=PartialProcess;}
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
    Task(PartialOT,v,sources, dests,edges, (float)sources.size()-0.1), dists(dists), offset(offset),
    QUANTA(QUANTA), index(index), sharedCnt(shared){this->type=PartialOT;}

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

std::atomic<long>  numProcessedVertex{0}, numProcessedEdge{0};
constexpr double EPS=1e-4;

 void calcCurvature(double alpha, Vertex v, vector<vector<double>> &MatDist, set<Edge> &edges, vector<Vertex> sources,
                    vector<Vertex> dests, Graph_t &g){

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
        int cntD = 0, destIndex = 0;
        for (Vertex dest: localDests) {
            auto range = equal_range(dests.begin(), dests.end(), dest);
            int indexj = range.first - dests.begin();
            for (int j = 0; j < MatDist.size(); j++) {
                vertDist[j][cntD] = MatDist[j][indexj];
                if (MatDist[j][indexj]==std::numeric_limits<double>::infinity())
                    int KKKK=0;
            }
            if (dest == neighbor)
                destIndex = cntD;
            cntD++;
        }

        int sA = vertDist.size(), sB = vertDist[0].size();
        //if (sB > 1) {

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

        wd = compute_EMD(distributionA, distributionB, vertDist);
        if (isnan(wd)) {
            g[e].ot = 9999.0;
            g[e].curv = 9999.0;
            cout<<"PROBLEM"<<endl;
        } else {
            g[e].ot = wd;
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

std::atomic<long> numShortestPath{0};
std::atomic<float> seuil={0.02};
auto t1=high_resolution_clock::now();
void process(int threadIndex, Graph_t *g, DistanceCache *distanceCache, TaskPriorityQueue *tasksToDo,
             long numbOfShortPaths, atomic<int> *runningTaskCount) {
    GraphNodeFactory graphNodeFact;
    Task *task;
    long totalTime1 = 0, totalTime2 = 0;
    int QUANTA=120;
    long numberShortestPath=0;
    float perc=0.0;
    seuil=0.02;
    auto t2=t1;
    do{
        while (tasksToDo->try_take(task,std::chrono::seconds(1000)) == BlockingCollectionStatus::Ok) {
            //        tasksToDo.try_take(task);
            numberShortestPath=0;
            Vertex s = task->v;
            Perf perf1, perf2;
            bool done = false;
            vector<Vertex> sources(task->sources), dests(task->dests);
            set<Edge> edges(task->edges);
            vector<vector<double>> *MatDist1;// *MatDist2;
            //        vector<vector<double>> *MatDist;

            int ssize = sources.size(), dsize = dests.size(), allsize=ssize*dsize;
            switch (task->type) {
                case FullProcess: {
                    FullTask *fullTask = (FullTask *) task;
                    (*runningTaskCount)++;
                    MatDist1 = new vector<vector<double>>(ssize, vector<double>(dsize,
                                                                                std::numeric_limits<double>::infinity()));
                    //                MatDist2= new vector<vector<double>>(ssize, vector<double>(dsize,std::numeric_limits<double>::infinity()));
                    if (ssize < QUANTA) {
                        perf1 = multisource_uniform_cost_search_seq1(MatDist1, 0, sources.size(),sources, dests, *g, *distanceCache, graphNodeFact);
                        std::atomic<int> *sharedCnt = new std::atomic<int>();
                        *sharedCnt = 0;
                        PartialCurvature *ptask = new PartialCurvature(sharedCnt, MatDist1, 0, edges.size(), fullTask->v, sources, dests, edges, 0);
                        tasksToDo->try_add((Task *) ptask);
                        //                    delete MatDist2;
                    } else {
                        std::atomic<int>  *sharedCnt = new std::atomic<int>();
                        *sharedCnt =0;
                        int iterNum = sources.size() / QUANTA;
                        auto begin = sources.begin();
                        for (int i = 0; i < sources.size(); i = i + QUANTA) {
                            PartialTask *ptask = new PartialTask(sharedCnt, MatDist1, i,std::min(QUANTA, (int)sources.size()-i),s,
                                                                 sources, dests, edges,*sharedCnt);
                            tasksToDo->try_add((Task *) ptask);
                            (*sharedCnt)++;
                        }
                    }
                    delete fullTask;
                    break;
                }
                case PartialProcess:{
                    PartialTask *partialTask = (PartialTask *) task;
                    //                t2 = high_resolution_clock::now();
//                cout<<"Partial Task:"<<partialTask->v<<":"<<*(partialTask->sharedCnt)<<", thread#:"<<threadIndex<<endl;
                    perf1 = multisource_uniform_cost_search_seq1(partialTask->dists, partialTask->offset, partialTask->QUANTA, sources, dests,
                                                                 *g,*distanceCache, graphNodeFact);
                    //                t3 = high_resolution_clock::now();
                    done = true;
                    (*(partialTask->sharedCnt))--;
                    if (*(partialTask->sharedCnt) == 0) {
                        delete partialTask->sharedCnt;
                        std::atomic<int> *sharedCnt = new std::atomic<int>();
                        *sharedCnt = 0;
                        int iterNum = edges.size() / QUANTA;
                        auto begin = edges.begin();
                        int i=0,j=0;
                        set<Edge> partialEdges;
                        for (Edge e:edges){
                            if (i==QUANTA){
                                PartialCurvature *ptask = new PartialCurvature(sharedCnt, partialTask->dists,j ,QUANTA, s, sources, dests,
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
                        if (partialEdges.size()>0){
                            PartialCurvature *ptask = new PartialCurvature(sharedCnt, partialTask->dists, j, partialEdges.size(), s, sources, dests,
                                                                           partialEdges,*sharedCnt);
                            tasksToDo->try_add((Task *) ptask);
                            partialEdges.clear();
                        }
                    }
                    delete partialTask;
                    break;
                }
                case PartialOT:{
                    PartialCurvature *partialOT = (PartialCurvature *) task;
                    //                t2 = high_resolution_clock::now();
                    calcCurvature(ALPHA,partialOT->v, *(partialOT->dists),partialOT->edges, sources,dests,*g);
                    done = true;
                    if (*(partialOT->sharedCnt) == 0) {
                        numberShortestPath=sources.size()*dests.size();
                        numProcessedVertex++;
                        done = true;
                        delete partialOT->dists;
                        delete partialOT->sharedCnt;
                        (*runningTaskCount)--;
                    } else {
                        (*(partialOT->sharedCnt))--;
                    }
                    delete partialOT;
                    break;
                }
            }
            numShortestPath+=numberShortestPath;
            perc=numShortestPath*1.0/numbOfShortPaths ;
            if (perc>seuil){
                seuil=seuil+0.02;
                t2 = high_resolution_clock::now();
                auto ms_double2 = duration_cast<microseconds>(t2 - t1);
                t1=t2;
                cout <<perc*100<< "%, numProcessedVertex=" << numProcessedVertex << ",numProcessedEdge=" << numProcessedEdge
                     << ",delay=" << ms_double2.count() << endl;
            }
        }
    } while(runningTaskCount!=0);
    t2 = high_resolution_clock::now();
    auto ms_double2 = duration_cast<microseconds>(t2 - t1);
    t1=t2;
    cout <<"End thread: "<< threadIndex<<","<<perc*100<< "%, numProcessedVertex=" << numProcessedVertex << ",numProcessedEdge=" << numProcessedEdge
         << ",delay=" << ms_double2.count() << endl;
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
    std::atomic<int> removed{0}, added{0};
    AdjacencyIterator nfirst, nend;
    VertexIterator v,vend;
    set<pair<Vertex, int>, vertexpaircomp> degreeSorted;
    long numbOfShortPaths=0;
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
        if (edgesToProcess.size() > 0) {
            numbOfEdges +=edgesToProcess.size();
            numbOfShortPaths +=sources.size()*destsSet.size();
            FullTask *task = new FullTask(src, sources, dests, edgesToProcess);
            tasksToDo.try_add((Task *) task);
        }
        sourcesSet.clear();
        destsSet.clear();
        edgesToProcess.clear();
    }
    cout<<"Number of tasks:"<<tasksToDo.size()<<" ,numbOfEdges="<<numbOfEdges<<", numbOfShortPaths="<<numbOfShortPaths<<endl;
    return numbOfShortPaths;
}

long generateTasks1(Graph_t &g, set<Task *, TaskPaircomp> &tasksToDoSet) {
    set<long> edgeSet;
    std::atomic<int> removed{0}, added{0};
    AdjacencyIterator nfirst, nend;
    VertexIterator v,vend;
    set<pair<Vertex, int>, vertexpaircomp> degreeSorted;
    long numbOfShortPaths=0;
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
            } else {}
        }
        vector<Vertex> sources(sourcesSet.begin(), sourcesSet.end()), dests(destsSet.begin(), destsSet.end());
        if (edgesToProcess.size() > 0) {
            numbOfEdges +=edgesToProcess.size();
            numbOfShortPaths +=sources.size()*destsSet.size();
            FullTask *task = new FullTask(src, sources, dests, edgesToProcess);
            tasksToDoSet.insert((Task *) task);
        }
        sourcesSet.clear();
        destsSet.clear();
        edgesToProcess.clear();
    }
    cout<<"Number of tasks:"<<tasksToDoSet.size()<<" ,numbOfEdges="<<numbOfEdges<<", numbOfShortPaths="<<numbOfShortPaths<<endl;
    return numbOfShortPaths;
}

struct Predicate {// both edge and vertex
    typedef typename graph_traits<Graph_t>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph_t>::edge_descriptor Edge;

    Predicate(Graph_t *g): g(g){};
    Predicate(){};
    bool operator()(Edge e) const      {return (*g)[e].active;}
    bool operator()(Vertex vd) const { return (*g)[vd].active; }
    Graph_t *g;
};

using Filtered_Graph_t = boost::filtered_graph<Graph_t, Predicate, Predicate>;

void k_core2(Graph_t &gin, Graph_t &gout, unsigned int k){
    VertexIterator v,vend;
    EdgeIterator e,eend;
    Graph_t *pgraph=new Graph_t();
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
    Edge maxEdge, minEdge;
    auto es = edges(g);
    for (auto eit = es.first; eit != es.second; ++eit) {
        Vertex src = source(*eit, g), dst = target(*eit, g);
        if (g[*eit].distance > maxdist) {
            maxdist = g[*eit].distance;
            maxEdge = *eit;
        }
        float tmpDist=g[*eit].distance;
        if (tmpDist < mindist) {
            mindist = g[*eit].distance;
        }
        if (isinf(g[*eit].curv)){
            cout<<"inf edge:"<<source(*eit,g)<<":"<<target(*eit,g)<<endl;
        }
        sumCurv += g[*eit].curv;
        int node = source(*eit, g);
        int comp = components[node];
        sumCurvperComponent[comp] = sumCurvperComponent[comp] + g[*eit].curv;
        sum2Curv += g[*eit].curv * g[*eit].curv;
        componentSize[comp]++;
        sum2CurvperComponent[comp] = sum2CurvperComponent[comp] + g[*eit].curv * g[*eit].curv;
        //        logFile1<<g[source(*eit,g)].name<<","<<g[target(*eit,g)].name<<","<<source(*eit,g)<<","<<target(*eit,g)<<","<<
        //g[*eit].distance << "," << g[*eit].curv << "," << g[*eit].ot << endl;
    }
    g[graph_bundle].avgCurv = sumCurv / num_edges(g);
    g[graph_bundle].stdCurv = sqrt(sum2Curv / num_edges(g) - g[graph_bundle].avgCurv * g[graph_bundle].avgCurv);
    cout << "maxdist:" << maxdist << ",mindist:" << mindist << ", maxEdge curv " << g[maxEdge].curv << endl;
    cout << "avgCurv=" << g[graph_bundle].avgCurv << ", stdCurv=" << g[graph_bundle].stdCurv << endl;
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
    for (auto eit = es.first; eit != es.second; ++eit) {
        //        double curv=1-g[*eit].ot/g[*eit].distance;
        double ddd = g[*eit].distance;
        g[*eit].distance = max(EPS, g[*eit].distance * (1 - g[*eit].curv));
        if ((g[*eit].distance <= EPS) && (!g[*eit].surgery)) { //we need a surgery of type 1
            surgery = true;
            g[*eit].surgery = true;
            g[*eit].distance = EPS;
            Vertex src = source(*eit, g), dst = target(*eit, g);
            cout << "Surgery Type 1: " << src << ":" << dst << ", Curvature:" << g[*eit].curv << endl;

            /*
            //            cout<<"Surgery Type 1: "<<src<<":"<<dst<<","<<g[src].name<<":"<<g[dst].name<<", Curvature:"<<g[*eit].curv<<endl;

            //            g[src].name=g[src].name+","+g[dst].name;
                    g[*eit].active=false;
                    g[dst].active=false;
                    for (auto ve = boost::out_edges(dst, g); ve.first != ve.second; ++ve.first) {
                        Vertex ddst = target(*ve.first, g);
                        if (ddst!=src){
                            auto ed=edge(src, ddst, g);
                            if (ed.second){
                                if (g[ed.first].distance>g[*ve.first].distance){
                                    g[ed.first].distance=g[*ve.first].distance;
                                }
                            } else {
                                add_edge(src, ddst, {g[*ve.first].dist, g[*ve.first].edist, g[*ve.first].distance, g[*ve.first].ot,
                                                     g[*ve.first].curv,
                                                     g[*ve.first].active}, g);

                                g[*ve.first].active = false;
                            }
                            if (g[*ve.first].distance<0)
                                int KKKK=0;
                        }
                    }*/

        } //else {
        sumWeights += g[*eit].distance;
        numEdgesUnfiltered++;
        //        }
    }
    int numV = num_vertices(g);
    int numE = num_edges(g);
    double rescaling = numEdgesUnfiltered * 1.0 / sumWeights;
    //    rescaling=1.0;
    //    rescaling=numE*1.0/sumWeights;
    oldrescaling = rescaling;

    //    rescaling=1.0;
    es = edges(g);
    for (auto eit = es.first; eit != es.second; ++eit) {
        double ddd = g[*eit].distance;
        Vertex src = source(*eit, g), dst = target(*eit, g);
        g[*eit].distance = g[*eit].distance * rescaling;
    }
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
            intersect.empty();
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

void ricci_flow(Graph_t *g, int numIteration, int iterationIndex, string path, dynamic_properties dpout) {
    Graph_t *ginter;
    double oldRescaling = 1.0;
    DistanceCache distanceCache(num_vertices(*g));
    vector<int> component(num_vertices(*g));
    int numComponent = connected_components(*g, &component[0]);
    ofstream logFile;
    ofstream outFile;
    atomic<int> runningTasksCount{0};
    auto t1 = high_resolution_clock::now(), t2 = t1;
//    long numbOfShortPaths=generateTasks(*g, tasksToDoSet);
    for (int index = iterationIndex; index < iterationIndex + numIteration; index++) {
        numShortestPath = 0;
        t1 = high_resolution_clock::now();
        TaskPriorityQueue tasksToDo;
        long numbOfShortPaths=generateTasks(*g, tasksToDo);
        string logFilename = path + "/processed/logFile." + to_string(index + 1) + ".log";
        logFile.open(logFilename.c_str(), ofstream::out);
        string outFilename = path + "/processed/processed." + to_string(index+1) + ".graphml";
        outFile.open(outFilename.c_str(), ofstream::out);
        cout << "Index:" << index << " ";
        numProcessedVertex = 0;
        numProcessedEdge = 0;
        int num_core = std::thread::hardware_concurrency();
//int        num_core=1;
        int offset = 0;
        int k = 0;
//    num_core=1;
        vector<thread> threads(num_core);
        for (int i = 0; i < num_core; i++) {
            threads[i] = std::thread(process, i, g, &distanceCache, &tasksToDo, numbOfShortPaths, &runningTasksCount);
        }
        for (int i = 0; i < num_core; i++) {
            threads[i].join();
        }
        calcStats(*g, numComponent, component);
        write_graphml(outFile, *g, dpout, true);
        if (updateDistances(*g, oldRescaling)) {
            ginter = new Graph_t();
            Predicate predicate(g);
            Filtered_Graph_t fg(*g, predicate, predicate);
            copy_graph(fg, *ginter);
            g->clear();
            delete g;
            g = ginter;
        }
        logFile.close();
        outFile.close();
        t2 = high_resolution_clock::now();
        auto executionTime = duration_cast<microseconds>(t2 - t1);
        cout << "Execution Time=" << executionTime.count() << ", avg per node="
             << executionTime.count() * 1.0 / num_vertices(*g) << endl;
        cout << "avg per link=" << executionTime.count() * 1.0 / num_edges(*g) << ", avg per path="
             << executionTime.count() * 1.0 / numShortestPath << endl;
    }
}
#endif //CURVATURE_CURVATUREHANDLER_H
