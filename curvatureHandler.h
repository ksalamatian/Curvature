//
// Created by ksalamatian on 14/02/2023.
//

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
#include <queue>
#include "emd.h"
#include "BlockingCollection.h"

#define ALPHA 0.0




using namespace boost;
using namespace std;
using namespace code_machina;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::microseconds;


class DistanceCache;


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


template <class Graph> class GraphNodeArray{
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
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
    std::atomic<long> numb{0}, yi{0};

};


template <class Graph> struct Multisource_uniform_cost_search_seq1{

    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;

    struct myComp {
        constexpr bool operator()(
                GraphNode* const& a,
                GraphNode* const& b)
        const noexcept
        {
            return a->dist > b->dist;
        }
    };
    Perf operator()(vector<vector<double>> *ddists, int offset, vector<Vertex> &sources,
                    vector<Vertex> &dests, Graph &g, DistanceCache &distanceCache){
        ;
        // minimum cost upto
        // goal state from starting
        // state

        // insert the starting index
        // map to store visited node
        set<Vertex> landMarks;
        Perf perf;
        perf.sourceSize=sources.size();
        perf.destSize=dests.size();
        long hit=0, checked=0;
        set<Vertex> visited;
        set<Vertex> settledNodes;
        set<long> sourceVisited;
        vector<int> srcCount(sources.size(),0);

        GraphNodeFactory graphNodeFact;
        GraphNodeArray<Graph> graphNodeArray(num_vertices(g),sources,distanceCache);
        priority_queue<GraphNode*, vector<GraphNode*>, myComp > *costQueue, *bakCostQueue, *swap;
        costQueue=new priority_queue<GraphNode*, vector<GraphNode*>, myComp >();
        std::set<Vertex> seen;

        int counter=0;
        int numSettled=0;
        int destsCount=0;
        bool finish=false;
        Vertex v;
        GraphNode *graphNode;
        for(int i=0;i<sources.size();i++){
            v=sources[i];
//        graphNodeArray.checkSet(v,i,0);
//        graphNodeArray.settle(v,i);
            graphNode = graphNodeFact.get(v,i,0.0);
            costQueue->push(graphNode);
            hit ++;
            numSettled ++;
        }
        set<Vertex> ddests(dests.begin(), dests.end());
        long maxQueue=0;
        double currDist, toDist, newDist, oldDist;
        Vertex neighbor;
        for(Vertex t:dests){
            if (ddests.count(t)>0){
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
                    //settle pop values
                    if (!graphNodeArray.isSettled(p->v, p->source)){
                        p->settled=true;
                        graphNodeArray.checkSet(p->v,p->source,p->dist);
                        graphNodeArray.settle(p->v, p->source);
                        if (ddests.count(p->v)>0) {
                            srcCount[p->source]++;
                            if (srcCount[p->source]>= dests.size()) {
                                bakCostQueue=new priority_queue<GraphNode*, vector<GraphNode*>, myComp >();
                                while (!costQueue->empty()) {
                                    GraphNode *pp=costQueue->top();
                                    costQueue->pop();
                                    if (pp->source==p->source) {
                                        graphNodeFact.yield(pp);
                                    } else {
                                        bakCostQueue->push(pp);
                                    }
                                }
                                delete costQueue;
                                costQueue=bakCostQueue;
                                sourceVisited.insert(((long)p->v<<32)+p->source);
                            }
                        }
//                    distanceCache1.insert(sources[p->source], p->v, abs(p->dist));
                        numSettled++;
                    }
                    if (graphNodeArray.isFinished(p->v)) {
                        ddests.erase(p->v);
                        if (ddests.empty()) {//all dests are attained
                            int j=0;
                            for (Vertex d: dests) {
                                auto range = equal_range(dests.begin(), dests.end(), d);
                                for (int i = 0; i < sources.size(); i++) {
                                    (*ddists)[i][j] = abs(graphNodeArray.get(d, i));
                                    if ((*ddists)[i][j]  == std::numeric_limits<double>::infinity()) {
                                        int KKKK=0;
                                    }
                                }
                                ++j;
                            }
                            perf.maxQueue=maxQueue;
                            perf.counter=counter;
                            perf.cacheHitRate = hit*1.0/checked;
                            while(!costQueue->empty()){
                                GraphNode *p=costQueue->top();
                                graphNodeFact.yield(p);
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
                        if (p->v==205)
                            int KKKK=0;

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
                                costQueue->push(graphNodeFact.get(neighbor, p->source,
                                                                  graphNodeArray.get(neighbor,
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
                            (*ddists)[i][j] = graphNodeArray.get(d, i);
                            if ((*ddists)[i][j]  == std::numeric_limits<double>::infinity()) {
                                int KKKK=0;
                            }
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
        delete costQueue;
        return perf;
    }

};

enum TaskType {FullProcess, PartialProcess, PartialCurvature,SuperProcess};
template<class Graph> class Task{
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;

public:
    TaskType type;
    Vertex v;
    vector<Vertex> sources;
    vector<Vertex> dests;
    set<Edge> edges;
    Task(TaskType type, Vertex v, vector<Vertex> &sources, vector<Vertex> &dests, set<Edge> &edges):type(type), v(v), sources(sources),
                                                                                                    dests(dests), edges(edges){}
    ~Task(){
        sources.clear();
        dests.clear();
        edges.clear();
    }
};

template<class Graph> struct TaskPaircomp {
    // Comparator function
    int operator()(Task<Graph> *l, Task<Graph> *r) const
    {
        long  lsize=l->dests.size()*l->sources.size(), rsize=r->dests.size()*r->sources.size();
//        long  lsize=l->sources.size(), rsize=r->sources.size();

        if (lsize < rsize) {
            return 1;
        } else
        if (rsize < lsize) {
            return -1;
        } else {
            return 0;
        }
    }
};

template<class Graph> class FullTask: public Task<Graph>{
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;
public:
    FullTask(Vertex v, vector<Vertex> &sources, vector<Vertex> &dests, set<Edge> &edges):Task<Graph>(FullProcess,v,sources, dests, edges)
    {};
};

template <class Graph> class PartialTask: public Task<Graph>{
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;
public:
    std::atomic<int> *sharedCnt;
    vector<vector<double>> *dists;
    int offset;
    PartialTask(std::atomic<int> *shared, vector<vector<double>> *dists, int offset, Vertex v,
                vector<Vertex> sources, vector<Vertex> dests, set<Edge> edges):
            Task<Graph>(PartialProcess,v,sources, dests,edges), dists(dists), offset(offset), sharedCnt(shared){this->type=PartialProcess;}
};




template <class Graph> using TaskPriorityQueue = PriorityBlockingCollection<Task<Graph> *, PriorityContainer<Task<Graph> *, TaskPaircomp<Graph>>>;
//template <class Graph> using TaskPriorityQueue = PriorityBlockingCollection<Task *, PriorityContainer<Task<Graph>*, TaskPaircomp<Graph>>>;

template<class Graph> TaskPriorityQueue<Graph> tasksToDo;
std::atomic<long>  numProcessedVertex{0}, numProcessedEdge{0};
auto t1 = high_resolution_clock::now(), t2=t1,t3=t1;
double EPS= 1e-4;


template<class Graph> struct CalcCurvature{
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;
    typedef typename boost::graph_traits < Graph >::adjacency_iterator AdjacencyIterator;

    void operator()(double alpha, Vertex v, vector<vector<double>> &MatDist, set<Edge> &edges, vector<Vertex> sources, vector<Vertex> dests, Graph &g){
        vector<double> distributionA;
        vector<double> distributionB;
        double wd;
        for (Edge e: edges) {
            Vertex src = source(e, g);
            Vertex neighbor = target(e, g);
            vector<vector<double>> vertDist(MatDist.size(), vector<double>(out_degree(neighbor, g) + 1,
                                                                           std::numeric_limits<double>::infinity()));
            pair<AdjacencyIterator, AdjacencyIterator> nB = adjacent_vertices(neighbor, g);
            set<Vertex> localDests;
            localDests.insert(neighbor);
            for (; nB.first != nB.second; localDests.insert(*nB.first++));
            int cntD = 0, destIndex = 0;
            for (Vertex dest: localDests) {
                auto range = equal_range(dests.begin(), dests.end(), dest);
                int indexj = range.first - dests.begin();
                for (int j = 0; j < MatDist.size(); j++) {
                    vertDist[j][cntD] = MatDist[j][indexj];
                }
                if (dest == neighbor)
                    destIndex = cntD;
                cntD++;
            }

            int sA = vertDist.size(), sB = vertDist[0].size();
            if (sB > 1) {
                double uA = (double) (1 - alpha) / (sA - 1);
                double uB = (double) (1 - alpha) / (sB - 1);
                distributionA.resize(sA);
                fill_n(distributionA.begin(), sA, uA);
                distributionB.resize(sB);
                fill_n(distributionB.begin(), sB, uB);
                //        vector<double> distributionA(sA, uA);
                //        vector<double> distributionB(sB, uB);
                auto range = equal_range(sources.begin(), sources.end(), src);
                int index = range.first - sources.begin();

                distributionA[index] = alpha;
                distributionB[destIndex] = alpha;

                wd = compute_EMD(distributionA, distributionB, vertDist);

                //            cout<<src<<":"<<wd<<":"<<neighbor<<endl;
                if (isnan(wd)) {
                    g[e].ot = 9999.0;
                    g[e].curv = 9999.0;
                } else {
                    g[e].ot = wd;
                    if (g[e].distance < EPS) {
                        g[e].curv = g[graph_bundle].avgCurv;
                    } else {
                        g[e].curv = 1 - wd / g[e].distance;
                        if (g[e].curv < -3) {
                            int KKKK = 0;
                        }
                    }
                }
                distributionA.clear();
                distributionB.clear();
            } else {
                numProcessedEdge++;
                g[e].curv = 0.0;
                g[e].ot = 0.0;
                g[e].distance = EPS / 2;
            }
            numProcessedEdge++;
        }
    }
};


template <class Graph> struct Process {

    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;

    void operator()(int threadIndex, Graph *g, DistanceCache *distanceCache) {
        Task<Graph> *task;

        CalcCurvature<Graph> calcCurvature;
        long totalTime1 = 0, totalTime2 = 0;
        int QUANTA=60;
        while (tasksToDo<Graph>.try_take(task) == BlockingCollectionStatus::Ok) {
            //        tasksToDo.try_take(task);
            Vertex s = task->v;
            Perf perf1, perf2;
            bool done = false;
            vector<Vertex> sources(task->sources), dests(task->dests);
            set<Edge> edges(task->edges);
            vector<vector<double>> *MatDist1;// *MatDist2;
            //        vector<vector<double>> *MatDist;
            int ssize = sources.size(), dsize = dests.size();
            switch (task->type) {
                case FullProcess: {
                    FullTask<Graph> *fullTask = (FullTask<Graph> *) task;
                    MatDist1 = new vector<vector<double>>(ssize, vector<double>(dsize,
                                                                                std::numeric_limits<double>::infinity()));
                    //                MatDist2= new vector<vector<double>>(ssize, vector<double>(dsize,std::numeric_limits<double>::infinity()));
                    if (ssize < QUANTA) {
                        Multisource_uniform_cost_search_seq1<Graph> multisource_uniform_cost_search_seq1;
                        perf1 = multisource_uniform_cost_search_seq1(MatDist1, 0, sources, dests, *g, *distanceCache);
                        //                    t2 = high_resolution_clock::now();
                        //                    perf2 = multisource_uniform_cost_search_seq(MatDist2, 0, sources, dests, *g);
                        //check for InF
                        //                    for (int i=0;i<ssize;i++){
                        //                        for (int j=0;j<dsize;j++) {
                        //                            if (isinf((*MatDist1)[i][j])) {
                        //                                multisource_uniform_cost_search_seq(MatDist2, 0, sources, dests, *g);
                        //                                int KKKK=0;
                        //                            }
                        //                            if ((*MatDist1)[i][j] != (*MatDist2)[i][j])
                        //                                cout << "NOT EQUAL  " << i << "," <<j << ","<<(*MatDist1)[i][j]<< ","<<(*MatDist2)[i][j]
                        //                                <<","<<(*MatDist1)[i][j]-(*MatDist2)[i][j]<<endl;
                        //                        }
                        //                   }

                        //                    t3 = high_resolution_clock::now()  ;
                        //                    t2 = high_resolution_clock::now();
                        //                    for(int i=0; i<ssize; i++){
                        //                        for (int j; j<dsize; j++){
                        //                            if ((*MatDist1)[i][j]!=(*MatDist2)[i][j])
                        //                                cout<<"NOT EQUAL "<<i<<","<<j<<endl;
                        //                        }
                        //                    }
                        //                    if ((*MatDist1)[0][0]==std::numeric_limits<double>::infinity())
                        //                        int KKKK=0;
                        calcCurvature(ALPHA, fullTask->v, *MatDist1, edges, sources, dests, *g);
                        delete MatDist1;
                        //                    delete MatDist2;
                        numProcessedVertex++;
                        done = true;
                    } else {
                        std::atomic<int> *sharedCnt = new std::atomic<int>();
                        *sharedCnt = 0;
                        int iterNum = sources.size() / QUANTA;
                        auto begin = sources.begin();
                        for (int i = 0; i < sources.size(); i = i + QUANTA) {
                            auto first = sources.begin() + i;
                            auto end = sources.begin() + i + std::min((long) QUANTA, (long) sources.size() - i);
                            vector<Vertex> partialVect(first, end);
                            PartialTask<Graph> *ptask = new PartialTask<Graph>(sharedCnt, MatDist1, i, s, partialVect, dests, edges);
                            tasksToDo<Graph>.try_add((Task<Graph> *) ptask);
                            (*sharedCnt)++;
                        }
                    }
                    delete fullTask;
                    break;
                }
                case PartialProcess:
                    PartialTask<Graph> *partialTask = (PartialTask<Graph> *) task;
                    cout << "PARTIAL TASK:" << partialTask->v << "," << *(partialTask->sharedCnt) << endl;
                    //                t2 = high_resolution_clock::now();
                    Multisource_uniform_cost_search_seq1<Graph> multisource_uniform_cost_search_seq1;
                    perf1 = multisource_uniform_cost_search_seq1(partialTask->dists, partialTask->offset, sources, dests,
                                                                *g,*distanceCache);
                    //                t3 = high_resolution_clock::now();
                    (*(partialTask->sharedCnt))--;
                    done = true;
                    if (*(partialTask->sharedCnt) == 0) {
                        calcCurvature(ALPHA, partialTask->v, *(partialTask->dists), partialTask->edges,
                                      partialTask->sources, partialTask->dests, *g);
                        delete partialTask->dists;
                        delete partialTask->sharedCnt;
                        numProcessedVertex++;
                    }
                    delete partialTask;
                    break;
            }
            //        if (done){
            //            ms_double1 = t2 - t1;
            //            ms_double2 = t3 - t1;
            //            totalTime1 += ms_double1.count();
            //            totalTime2 += ms_double2.count();
            //            cout << s << "," << threadIndex << "," << numProcessedVertex << "," << numProcessedEdge <<","<<visNodeFact.number()
            //                 << endl;
            //            cout << s << "," <<totalTime1<<"," << ms_double1.count() << "," << sources.size() << "," << dests.size() <<
            //               "," << perf1.maxQueue << "," << perf1.counter << "," << perf1.cacheHitRate << endl;
            //            cout << s << "," <<totalTime2<< "," << ms_double2.count() << "," << sources.size() << "," << dests.size() <<
            //                "," << perf2.maxQueue << "," << perf2.counter<<"," <<ms_double2.count()*1.0/(sources.size()*dests.size())<< endl;
            //            logFile << s << "," << threadIndex << "," << numProcessedVertex << "," << numProcessedEdge << ","
            //                    << visNodeFact.number() << "," << ms_double2.count() << "," << sources.size() << "," << dests.size() << ","
            //                    << perf2.maxQueue << "," << perf2.counter << "," << ms_double2.count()*1.0/(sources.size()*dests.size())<<endl;
            //        }
            if (numProcessedVertex % 5000 == 0) {
                t2 = high_resolution_clock::now();
                auto ms_double2 = duration_cast<microseconds>(t2 - t1);
                cout << "numProcessedVertex=" << numProcessedVertex << ",numProcessedEdge=" << numProcessedEdge
                     << ",delay=" << ms_double2.count() << endl;
                //        ofstream preFile;
                //        preFile.open("/data/Curvature/processed" + to_string(numProcessedVertex) + ".graphml", ofstream::out);
                //        boost::dynamic_properties dpout;
                //        write_graphml(preFile, *g, dpout, true);
                //
            }
        }
    }
};

template<class Graph> struct vertexpaircomp {
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;

    // Comparator function
    bool operator()(const pair<Vertex, int>& l, const pair<Vertex, int>& r) const
    {
        if (l.second != r.second) {
            return l.second < r.second;
        }
        return true;
    }
};

template <class Graph> struct GenerateTasks {
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;

    void operator()(Graph &g, TaskPriorityQueue<Graph> &tasksToDo) {
        typename Graph::vertex_iterator v, vend;
        set<long> edgeSet;
        std::atomic<int> removed{0}, added{0};

        set<pair<Vertex, int>, vertexpaircomp<Graph>> degreeSorted;
        for (tie(v, vend) = vertices(g); v != vend; v++) {
            degreeSorted.insert(make_pair(*v, degree(*v, g)));
        }
/*    int k=0, offset, num_core=std::thread::hardware_concurrency();
    int scale=sortedVertices.size()/num_core;
    for (auto elem:degreeSorted){
        offset=(k%num_core)*scale+k/num_core;
        sortedVertices[offset]=elem.first;
        k++;
    }*/
//    parallel_for(blocked_range<int>(0,sortedVertices.size(),50000),
//                 [&](blocked_range<int> r) {
        typename Graph::adjacency_iterator nfirst, nend;
        set<Vertex> sourcesSet, destsSet;
        set<Task<Graph> *, TaskPaircomp<Graph>> taskSet;
        set<Edge> edgesToProcess;
        for (auto p: degreeSorted) {
//            for (int i=r.begin(); i<r.end(); ++i) {
            Vertex src = p.first;
            sourcesSet.insert(src);
            for (auto ve = boost::out_edges(src, g); ve.first != ve.second; ++ve.first) {
                bool found = true;
                Vertex dest = target(*ve.first, g);
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
            vector<Vertex> sources(sourcesSet.begin(), sourcesSet.end()), dests(destsSet.begin(), destsSet.end());
            if (edgesToProcess.size() > 0) {
                FullTask<Graph> *task = new FullTask<Graph>(src, sources, dests, edgesToProcess);
                tasksToDo.try_add((Task<Graph> *) task);
            }
            sourcesSet.clear();
            destsSet.clear();
            edgesToProcess.clear();
        }
//                 });
    }
};

template<class Graph> struct Predicate {// both edge and vertex
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;

    Predicate(Graph *g): g(g){};
    Predicate(){};
    bool operator()(Edge e) const      {return (*g)[e].active;}
    bool operator()(Vertex vd) const { return (*g)[vd].active; }
    Graph *g;
};

template<class Graph> using Filtered_Graph_t = boost::filtered_graph<Graph, Predicate<Graph>, Predicate<Graph>>;

template <class Graph> struct K_core2{
    typedef typename boost::graph_traits < Graph >::vertex_iterator VertexIterator;
    typedef typename boost::graph_traits < Graph >::edge_iterator EdgeIterator;
    typedef typename boost::graph_traits < Graph >::vertex_descriptor Vertex;
    typedef typename boost::graph_traits < Graph >::edge_descriptor Edge;
    typedef typename boost::property_map < Graph, boost::vertex_index_t >::type IndexMap;

    void operator()(Graph &gin, Graph &gout, unsigned int k) const {
        VertexIterator v,vend;
        EdgeIterator e,eend;
        Graph *pgraph=new Graph();
        vector<unsigned long> degrees(num_vertices(gin));
        map<Edge,bool> allEdges;
        set<pair<Vertex, int>, vertexpaircomp<Graph>> degreeSorted;
        copy_graph(gin,*pgraph);
        bool proceed;
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
            Predicate<Graph> predicate(pgraph);
            Filtered_Graph_t<Graph> fg(*pgraph, predicate, predicate);
            Graph *toDelete=pgraph;
            pgraph=new Graph();
            copy_graph(fg,*pgraph);
            toDelete->clear();
            delete toDelete;
        } while(proceed);
        copy_graph(*pgraph,gout);
        cout<<"num vertices: "<<num_vertices(gout)<<", num edges: "<<num_edges(gout)<<endl;
        delete pgraph;
    }

};


template<class Graph> class SuperTask: public Task<Graph>{
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;

    vector<Task<Graph>> subtasks;
    SuperTask(vector<Task<Graph>> &tasks, vector<Vertex> &superSources, vector <Vertex> &superDests, set<Edge> &superEdges):
            Task<Graph>(SuperProcess,0,superSources, superDests, superEdges), subtasks(tasks){}
};


template<class Graph> struct CalcStats {
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;
    typedef typename boost::graph_traits<Graph>::adjacency_iterator AdjacencyIterator;

    void operator()(Graph &g, int componentNum, vector<int> &components) {
        double maxdist = -1.0, mindist = 999999.0;
        vector<double> sumCurvperComponent(componentNum, 0.0), sum2CurvperComponent(componentNum, 0.0);
        vector<int> componentSize(componentNum, 0);
        double sumCurv = 0.0, sum2Curv = 0.0;
        Edge maxEdge, minEdge;
        auto es = edges(g);
        for (auto eit = es.first; eit != es.second; ++eit) {
            double ddd = g[*eit].distance;
            Vertex src = source(*eit, g), dst = target(*eit, g);
            if (g[*eit].distance > maxdist) {
                maxdist = g[*eit].distance;
                maxEdge = *eit;
            }
            if (g[*eit].distance < mindist) {
                mindist = g[*eit].distance;
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
};

template<class Graph> struct UpdateDistances {
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;

    bool operator()(Graph &g, double &oldrescaling) {
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
};

template <class Graph> struct Triangle_checker{
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph>::edge_descriptor Edge;
    typedef typename boost::graph_traits < Graph >::vertex_iterator VertexIterator;
    typedef typename boost::graph_traits < Graph >::edge_iterator EdgeIterator;
    typedef typename boost::graph_traits<Graph>::adjacency_iterator AdjacencyIterator;

    bool operator()(Graph &g,int type){
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
};


#endif //CURVATURE_CURVATUREHANDLER_H
