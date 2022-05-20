//
// Created by Kave Salamatian on 20/04/2021.
//

#include <stdio.h>
#include <iostream>
#include <utility>
#include <thread>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphml.hpp>
//#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/copy.hpp>
#include <boost/multi_index/composite_key.hpp>
#include "emd.h"
#include <tbb/parallel_for.h>
#include "BlockingCollection.h"

#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_queue.h>
#include <queue>
#include "LruCache.h"

using namespace boost;
using namespace std;
using namespace tbb;
using namespace code_machina;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

struct Vertex_info;
struct Edge_info;
typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::undirectedS, Vertex_info, Edge_info > Graph_t;
typedef boost::graph_traits < Graph_t >::vertex_iterator VertexIterator;
typedef boost::graph_traits < Graph_t >::edge_iterator EdgeIterator;
typedef boost::graph_traits < Graph_t >::adjacency_iterator AdjacencyIterator;
typedef boost::graph_traits < Graph_t >::vertex_descriptor Vertex;
typedef boost::graph_traits < Graph_t >::edge_descriptor Edge;
typedef boost::property_map < Graph_t, boost::vertex_index_t >::type IndexMap;

struct Vertex_info {
    std::string country;
    std::string name;
    int prefixnum;
    int prefixall;
    int astime;
    int addall;
    int addnum;
    std::string asnumber;
    int pathnum;
    double potential;
};

struct Edge_info {
    int weight= 1;
    long pathcount;
    int edgetime;
    //int ctime;
    int prefcount;
    double distance=1.0;
    double ot=99.0;
    bool visited=false;
};

bool POTENTIAL=false;
int QUANTA=30;
ofstream logFile, logFile1;
Graph_t g;
boost::dynamic_properties dp;

/*
 * Read a graphml
 */
void readGraphMLFile (Graph_t& designG, std::string fileName ) {

    ifstream inFile;
    dp.property("asNumber", get(&Vertex_info::asnumber, designG));
    //dp.property("pathnum", get(&Vertex_info::pathnum, designG));
    dp.property("Country", get(&Vertex_info::country, designG));
    dp.property("Name", get(&Vertex_info::name, designG));
    dp.property("asTime", get(&Vertex_info::astime, designG));
    dp.property("prefixNum", get(&Vertex_info::prefixnum, designG));
    dp.property("prefixAll", get(&Vertex_info::prefixall, designG));
    // dp.property("addAll", get(&Vertex_info::addall, designG));
    // dp.property("addNum", get(&Vertex_info::addnum, designG));
    dp.property("count", get(&Edge_info::pathcount, designG));
    dp.property("edgeTime", get(&Edge_info::edgetime, designG));
    dp.property("weight", get(&Edge_info::weight, designG));
    dp.property("distance", get(&Edge_info::distance, designG));
    dp.property("ot", get(&Edge_info::ot, designG));
    inFile.open(fileName, ifstream::in);
    try {
        boost::read_graphml(inFile, designG, dp);
    }
    catch (const std::exception &exc) {
        cerr << exc.what();
        cout << "Type: " << typeid(exc).name() << "\n";
    }
    cout << "Num Vertices: " << num_vertices(designG) << endl;
    cout << "Num Edges: " << num_edges(designG) << endl;
    inFile.close();
}

void k_core(Graph_t &gin, Graph_t &gout, int k){
    Graph_t *before, empty;
    before= &gin;
    Graph_t::vertex_descriptor vsource, vsource1,vdest, vdest1;
    std::pair<Graph_t::edge_descriptor, bool> e;
    cout<<"k-core 0 size is:"<<num_vertices(gin)<<","<<num_edges(gin)<<endl;
    for(int i=0;i<k;i++) {
        Graph_t *after, empty;
        after=&empty;
        vector<int> color(num_vertices(*before), 0);
        bool sourceOK, destOK;
        vector<Vertex> indices(num_vertices(*before),1000000);
        auto es = edges(*before);
        for (auto eit = es.first; eit != es.second; ++eit) {
            sourceOK=false;
            destOK=false;
            vsource1 = source(*eit, *before);
            if (color[vsource1] == 0) {
                if (degree(vsource1, *before) > 1) {
                    vsource = add_vertex(*after);
                    indices[vsource1]=vsource;
                    (*after)[vsource].name = (*before)[vsource1].name;
                    (*after)[vsource].country = (*before)[vsource1].country;
                    (*after)[vsource].astime = (*before)[vsource1].astime;
                    (*after)[vsource].pathnum = (*before)[vsource1].pathnum;
                    (*after)[vsource].prefixnum = (*before)[vsource1].prefixnum;
                    (*after)[vsource].prefixall = (*before)[vsource1].prefixall;
                    (*after)[vsource].asnumber = (*before)[vsource1].asnumber;
                    sourceOK=true;
                    color[vsource1]= 1;
                }
            } else {
                vsource= indices[vsource1];
                sourceOK=true;
            }
            vdest1 = target(*eit, *before);
            if (vdest1 != vsource1) {
                if (color[vdest1] == 0) {
                    if (degree(vdest1, *before) > 1) {
                        vdest = add_vertex(*after);
                        indices[vdest1]=vdest;
                        (*after)[vdest].name = (*before)[vdest1].name;
                        (*after)[vdest].country = (*before)[vdest1].country;
                        (*after)[vdest].astime = (*before)[vdest1].astime;
                        (*after)[vdest].pathnum = (*before)[vdest1].pathnum;
                        (*after)[vdest].prefixnum = (*before)[vdest1].prefixnum;
                        (*after)[vdest].prefixall = (*before)[vdest1].prefixall;
                        (*after)[vdest].asnumber = (*before)[vdest1].asnumber;
                        destOK = true;
                        color[vdest1] = 1;
                    }
                } else {
                    vdest=indices[vdest1];
                    destOK = true;
                }
            }
            if (sourceOK && destOK){
                e = add_edge(vsource, vdest, *after);
                (*after)[e.first].weight = (*before)[*eit].weight;
                (*after)[e.first].pathcount = (*before)[*eit].pathcount;
                (*after)[e.first].edgetime = (*before)[*eit].edgetime;
                (*after)[e.first].prefcount = (*before)[*eit].prefcount;
                (*after)[e.first].distance = (*before)[*eit].distance;
            }
        }
        cout<<"k-core "<<i+1<<" size is:"<<num_vertices(*after)<<","<<num_edges(*after)<<endl;
        before->clear();
        copy_graph(*after,*before);
        indices.clear();
    }
    copy_graph(*before,gout);
}



enum TaskType {FullProcess, PartialProcess, PartialCurvature};
class Task{
public:
    TaskType type;
    Vertex v;
    vector<Vertex> sources;
    vector<Vertex> dests;
    set<Edge> edges;
    Task(TaskType type, Vertex v, vector<Vertex> sources, vector<Vertex> dests, set<Edge> edges):type(type), v(v), sources(sources),
    dests(dests), edges(edges){}
};

class FullTask: public Task{
public:
    FullTask(Vertex v, vector<Vertex> &sources, vector<Vertex> &dests, set<Edge> edges):Task(FullProcess,v,sources, dests, edges)
    {};
};

class PartialTask: public Task{
public:
    std::atomic<int> *sharedCnt;
    vector<vector<float>> *dists;
    int offset;
    PartialTask(std::atomic<int> *shared, vector<vector<float>> *dists, int offset, Vertex v, vector<Vertex> sources, vector<Vertex> dests, set<Edge> edges):
    Task(PartialProcess,v,sources, dests,edges), dists(dists), offset(offset), sharedCnt(shared){type=PartialProcess;}
};

struct TaskPaircomp {
    // Comparator function
    int operator()(Task *l, Task *r) const
    {
//        long  lsize=l->dests.size()*l->sources.size(), rsize=r->dests.size()*r->sources.size();
        long  lsize=l->sources.size(), rsize=r->sources.size();

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

using TaskPriorityQueue = PriorityBlockingCollection<Task *, PriorityContainer<Task *, TaskPaircomp>>;

TaskPriorityQueue tasksToDo;

void generateTasks1(Graph_t& g, TaskPriorityQueue &tasksToDo){
    Graph_t::edge_iterator e, eend;
    map<Edge,bool> allEdges;

    for (tie(e, eend) = edges(g); e != eend; allEdges[*e++]=false);
    int count=0;
    for (auto ed: allEdges){
        if (!ed.second){
            Vertex t=target(ed.first,g), s=source(ed.first,g), v, w, tt,ss;
            set<Vertex> sourceNeighborsSet, destNeighborsSet;
            set<Edge> edgesToProcess;
            if (degree(t,g)>degree(s,g)){
                v=s;
                w=t;
            } else {
                v=t;
                w=s;
            }
            // we should also add all neighbors links of v
            for ( auto ve = boost::out_edges(v, g ); ve.first != ve.second; ++ve.first){
                Graph_t::adjacency_iterator nfirst, nend;
                tt=target(*ve.first,g);
                ss=source(*ve.first,g);
                if (ss!=v)
                    int KKKK=0;
                sourceNeighborsSet.insert(tt);
                destNeighborsSet.insert(tt);
                if (!allEdges[*ve.first]){
                    edgesToProcess.insert(*ve.first);
                    allEdges[*ve.first]=true;
                    count++;
                    tie(nfirst, nend) = adjacent_vertices(tt, g);
                    destNeighborsSet.insert(nfirst,nend);
                }
            }
            if (destNeighborsSet.count(w)==0){
                cout<<"BUG:"<<destNeighborsSet.size()<<","<<w<<endl;
                int KKKK=0;
            }
            vector<Vertex> sources(sourceNeighborsSet.begin(),sourceNeighborsSet.end()),
                    dests(destNeighborsSet.begin(), destNeighborsSet.end());
            if (destNeighborsSet.size()>0){
                FullTask *task = new FullTask(v, sources, dests, edgesToProcess);
                tasksToDo.try_add((Task *)task);
            }
            cout<<count<<endl;
        }
    }
    int KKKK=0;
}

void generateTasks(Graph_t& g, TaskPriorityQueue &tasksToDo){
    Graph_t::vertex_iterator v, vend;
    concurrent_unordered_set<long> edgeSet;
    std::atomic<int> removed{0}, added{0};

    set<pair<Vertex, int>, vertexpaircomp> degreeSorted;
    for (tie(v, vend) = vertices(g); v != vend; v++){
        degreeSorted.insert(make_pair(*v,degree(*v,g)));
    }
    vector<Vertex> sortedVertices(degreeSorted.size());
    for(auto it=degreeSorted.begin();it!=degreeSorted.end();sortedVertices.push_back((*it).first))
/*    int k=0, offset, num_core=std::thread::hardware_concurrency();
    int scale=sortedVertices.size()/num_core;
    for (auto elem:degreeSorted){
        offset=(k%num_core)*scale+k/num_core;
        sortedVertices[offset]=elem.first;
        k++;
    }*/
    parallel_for(blocked_range<int>(0,sortedVertices.size(),50000),
                 [&](blocked_range<int> r) {
                     Graph_t::adjacency_iterator nfirst, nend;
                     set<Vertex> sourcesSet, destsSet;
                     set<Task *, TaskPaircomp> taskSet;
                     set<Edge> edgesToProcess;
                     for (int i=r.begin(); i<r.end(); ++i) {
                         Vertex v=sortedVertices[i];
                         for ( auto ve = boost::out_edges(v, g ); ve.first != ve.second; ++ve.first){
                             Vertex d=target(*ve.first,g);
                             Vertex s=source(*ve.first,g);
                             sourcesSet.insert(d);
                             long key=(s<<32)+d;
                             if (edgeSet.find(key) == edgeSet.end()) {
                                 key=(d<<32)+s;
                             }
                             auto it=edgeSet.insert(key);
                             if (it.second){
                                 // the edge have not been processed
                                 tie(nfirst, nend) = adjacent_vertices(d, g);
                                 destsSet.insert(nfirst,nend);
                                 edgesToProcess.insert(*ve.first);
                                 added++;
                             } else {
                                 removed ++;
                             }
                         }
                         vector<Vertex> sources(sourcesSet.begin(),sourcesSet.end()), dests(destsSet.begin(),destsSet.end());
                         if (destsSet.size()>0){
                            FullTask *task = new FullTask(v, sources, dests, edgesToProcess);
                             tasksToDo.try_add((Task *)task);
                         }
                         sourcesSet.clear();
                         destsSet.clear();
                         edgesToProcess.clear();
                     }
                 });
}

/*
 * To limite the visited edge in the dijkstra algorithm
 */
class My_visitor : boost::default_bfs_visitor{
protected:
    set<Vertex> dests;
public:
    My_visitor( const set<Vertex> destset)
    : dests(destset) {};
    void initialize_vertex(const Vertex &s, const Graph_t &g) const {}
    void discover_vertex(const Vertex &s, const Graph_t &g) const {}
    void examine_vertex(const Vertex &s, const Graph_t &g) const {}
    void examine_edge(const Edge &e, const Graph_t &g) const {}
    void edge_relaxed(const Edge &e, const Graph_t &g) const {}
    void edge_not_relaxed(const Edge &e, const Graph_t &g) const {}
    void finish_vertex(const Vertex &s, const Graph_t &g) {
        dests.erase(s);
        if (dests.empty())
            throw(2);
    }
};

class IntNode {
public:
    Vertex v;
    double distance=std::numeric_limits<double>::infinity(); //distance from source v
    bool visited=false; //this node has been visited by source v
    IntNode(){};
    IntNode(Vertex v,double dist): v(v),distance(dist){};
    IntNode(const IntNode &node): v(node.v), distance(node.distance){}
    struct distChange : public std::unary_function<IntNode,void> {
        double d; distChange(const double &_d) : d(_d) {}
        void operator()(IntNode *node) { node->distance = d;}
    };
};

struct IntNodeByV {};
struct IntNodeByD {};
using InternalHeap=  boost::multi_index_container< IntNode*, // the data type stored
boost::multi_index::indexed_by< // list of indexes
boost::multi_index::ordered_unique<  //hashed index over Vertex
boost::multi_index::tag<IntNodeByV>, // give that index a name
boost::multi_index::member<IntNode, Vertex, &IntNode::v> // what will be the index's key
>,
boost::multi_index::ordered_non_unique<  //hashed non-unique index over distance vector
boost::multi_index::tag<IntNodeByD>, // give that index a name
boost::multi_index::member<IntNode, double, &IntNode::distance > // what will be the index's key
>
>
>;


class DistanceCache{
private:
    ThreadSafeScalableCache<long, double> distanceCache;
public:
    int cacheSize;
    std::atomic<long> hit{0};
    std::atomic<long> miss{0};
    DistanceCache(size_t maxSize, size_t numShards):distanceCache(maxSize, numShards), cacheSize(maxSize){}
    ~DistanceCache(){
        distanceCache.clear();
//        delete &distanceCache;
    }
    pair<bool, short int> insert(const Vertex src, const Vertex dst, double value){
        long key= src;
        key=(key<<32)+dst;
        return distanceCache.insert(key,value);
    }

    double find(const Vertex src, const Vertex dst) {
        // return the distance if he finds it and -1 if not
        ThreadSafeScalableCache<long, double>::ConstAccessor ac;
        long key= src;
        key=(key<<32)+dst;
        if (distanceCache.find(ac,key)){
            hit++;
            return *ac;
        }
        key=dst;
        key=(key<<32)+src;
        if (distanceCache.find(ac,key)){
            hit++;
            return *ac;
        }
        miss++;
        return -1;
    }

    int size(){
        return distanceCache.size();
    }

    void clear(){
        distanceCache.clear();
    }
};


struct VisNodeByV {};
struct VisNodeByD {};
class VisNode {
private:
    double potential=0.0;
    InternalHeap sortedDistances;
public:
    double minDist;
    map<Vertex, IntNode> settled;
    set<Vertex> unprocessed;
    typename boost::multi_index::index<InternalHeap,IntNodeByV>::type& indexNodeByV = sortedDistances.get<IntNodeByV>(); // here's where index tags (=names) come in handy
    typename boost::multi_index::index<InternalHeap,IntNodeByD>::type& indexNodeByD = sortedDistances.get<IntNodeByD>();
    Vertex v;

    VisNode(Vertex v, Vertex target, vector<Vertex> &sources, DistanceCache &distanceCache): v(v){
        double dist, mdist=std::numeric_limits<double>::infinity();
        for(Vertex w:sources){
            if (v==w){
                indexNodeByD.insert(new IntNode(w,0));
            } else {
                dist=distanceCache.find(v,w);
                if  (dist>-1){
                    //indexNodeByD.insert(new IntNode(w,dist));
                    settled[w].distance=dist;
                    settled[w].v=w;
                    settled[w].visited=false;
                    unprocessed.insert(w);
                    mdist=min(mdist,dist);
                } else {
                    indexNodeByD.insert(new IntNode(w,std::numeric_limits<double>::infinity()));
                }
            }
        }
        if (POTENTIAL){
            dist=distanceCache.find(v,target);
            if (dist>-1){
                potential=dist;
            } else {
                potential=0.0;
            }
        }
        minDist=mdist+potential;
    }

    ~VisNode(){
        for (auto p:indexNodeByV){
            delete p;
        }
        sortedDistances.clear();
        settled.clear();
    }

    void empty(){
        for (auto p:indexNodeByV){
            delete p;
        }
        sortedDistances.clear();
        settled.clear();
        unprocessed.clear();
    }

    void fill(Vertex w, Vertex target, vector<Vertex> &ssources,  DistanceCache &ddistanceCache ){
        v=w;
        double dist, mdist=std::numeric_limits<double>::infinity();
        for(Vertex w:ssources){
            if (v==w){
//                indexNodeByD.insert(new IntNode(w,0));
                settled[w].distance=0;
                settled[w].v=w;
                settled[w].visited=false;
                unprocessed.insert(w);
                mdist=0;
            } else {
                dist=ddistanceCache.find(v,w);
                if  (dist>-1) {
//                    indexNodeByD.insert(new IntNode(w,dist));
                    settled[w].distance=dist;
                    settled[w].v=w;
                    settled[w].visited=false;
                    unprocessed.insert(w);
                    mdist=min(mdist,dist);
                } else {
                    indexNodeByD.insert(new IntNode(w,std::numeric_limits<double>::infinity()));
                }
            }
        }
        if (POTENTIAL){
            dist=ddistanceCache.find(v,target);
            if (dist>-1){
                potential=dist;
            } else {
                potential=0.0;
            }
        }
        minDist=mdist+potential;
    }

    double get_potential(Vertex node,Graph_t &g, set<Vertex> &landMarks, Vertex target, DistanceCache &distanceCache){
        double maxPotential=0.0;
        for(Vertex v:landMarks) {
            double dnode=distanceCache.find(v,node);
            double dtarget=distanceCache.find(v,target);
            if ((dnode>-1) && (dtarget>-1))
                maxPotential=max(maxPotential,abs(dnode-dtarget));
        }
        return maxPotential;
    }


    IntNode get_settledIntNode(Vertex w){
        return settled[w];
    }

    bool update_distance(Vertex w, double d){
        if (settled.count(w)==0){
            auto it=indexNodeByV.find(w);
            if ((*it)->distance>d){
                indexNodeByV.modify(it,IntNode::distChange(d));
                double minDistC=minDist;
                minDist=(*indexNodeByD.begin())->distance+potential;
                if (minDist<minDistC)
                    return true;
            }
        }
        return false;
    }

    void update_potential(Vertex target, Graph_t &g, set<Vertex> landMarks, DistanceCache &distanceCache){
        double dist=distanceCache.find(v,target);
        if (dist>-1){
            potential=dist;
        } else {
            potential=0.0;
        }
        if (indexNodeByV.size()>0)
            minDist=(*indexNodeByD.begin())->distance+potential;
        //        potential= get_potential(v,g, landMarks, target, distanceCache);
        //        if (indexNodeByV.size()>0)
        //            minDist=(*indexNodeByD.begin())->distance+potential;
        //        else
        //            minDist=potential;
    }

    double get_minDistance(){
        if (indexNodeByV.size()>0)
            return (*(indexNodeByD.begin()))->distance;
        else
            return std::numeric_limits<double>::infinity();
    }

    double get_minPotentialDistance(){
        return minDist;
    }

    IntNode get_headSource(){
        return **(indexNodeByD.begin());
    }
    struct update_minDist : public std::unary_function<VisNode,void> {
        double d; update_minDist(const double &_d) : d(_d) {}
        void operator()(VisNode *node) { node->minDist = d+node->potential;}
    };

    set<Vertex> settle_distances(double nextDist, DistanceCache &distanceCache) {
        //check if any distance can be settled
        set<Vertex> newSettled;
        auto it =indexNodeByD.begin();
        for(Vertex u:unprocessed){
            newSettled.insert(u);
        }
        unprocessed.clear();
        while (it != indexNodeByD.end()) {
            Vertex currentSource = (*it)->v;
            double currentDistance = (*it)->distance;
            if (currentDistance !=
            std::numeric_limits<double>::infinity()) {// node v has never been reached from currentSource
                if (currentDistance <= nextDist) {
                    newSettled.insert(currentSource);
                    settled[currentSource]=*(*it);
                    distanceCache.insert(currentSource, v,currentDistance);
                    delete *it;
                    it = indexNodeByD.erase(it);
                } else {
                    it++;
                }
            } else{
                break;
            }
        }
        if (!is_fullySettled()) {
            minDist=(*indexNodeByD.begin())->distance+potential;
        } else {
            minDist=std::numeric_limits<double>::infinity();
        }
        return newSettled;
    }

    bool is_fullySettled(){
        return (indexNodeByV.size()==0);
    }
};

struct Comp {
    bool operator()(const map<Vertex, IntNode> &a, const map<Vertex, IntNode> &b) const {
        return a.size() > b.size();
    }
};

using Heap = boost::multi_index_container< VisNode*, // the data type stored
boost::multi_index::indexed_by< // list of indexes
boost::multi_index::hashed_unique<  //hashed index over Vertex
boost::multi_index::tag<VisNodeByV>, // give that index a name
boost::multi_index::member<VisNode, Vertex, &VisNode::v> // what will be the index's key
>,
boost::multi_index::ordered_non_unique<  //hashed non-unique index over distance vector
boost::multi_index::tag<VisNodeByD>, // give that index a name
//boost::multi_index::member<VisNode, double, &VisNode::minDist > // what will be the index's key
boost::multi_index::composite_key<VisNode,
boost::multi_index::member<VisNode, double, &VisNode::minDist>,
boost::multi_index::member<VisNode, map<Vertex, IntNode>, &VisNode::settled>
>,
boost::multi_index::composite_key_compare<
std::less<double>,   //minDist order by normal order
Comp // size of settled
>
>
>
>;



class VisNodeFactory {
public:

    VisNodeFactory(){};
    ~VisNodeFactory(){
        VisNode *elem;
        while (!visNodeQueue.empty()){
            visNodeQueue.try_pop(elem);
            delete elem;
        }
    }

    VisNode *get(Vertex v, Vertex target, vector<Vertex> &sources, DistanceCache &distanceCache){
        VisNode *elem;
        if (!visNodeQueue.try_pop(elem)){
            elem=new VisNode(v,target, sources, distanceCache);
            numb++;
        } else {
            elem->fill(v,target, sources, distanceCache);
        }
        long s=visNodeQueue.unsafe_size();
        return elem;
    };

    void yield(VisNode *visNode){
        visNode->empty();
        visNodeQueue.push(visNode);
        yi++;
    }

    long number(){
        return numb;
    }

private:
    tbb::concurrent_queue<VisNode*> visNodeQueue;
    std::atomic<long> numb{0}, yi{0};

};

struct Perf{
public:
    long maxQueue=0.0;
    long counter = 0.0;
    double cacheHitRate=0.0;
    long sourceSize=0.0;
    long destSize=0.0;
};

//Global variables
DistanceCache distanceCache(20000000,48);
VisNodeFactory visNodeFact;

#define EPS 0.0
Perf multisource_uniform_cost_search_seq(vector<vector<float>> *ddists, int offset, vector<Vertex> &sources, vector<Vertex> &dests, Graph_t &g){
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
    // create a priority heap
    Heap heap;
    auto& indexByV = heap.get<VisNodeByV>(); // here's where index tags (=names) come in handy
    auto& indexByD = heap.get<VisNodeByD>();
    int counter=0;
    int numSettled=0;
    int destsCount=0;
    bool finish=false;
    set<Vertex> ddests(dests.begin(), dests.end());
    for(Vertex v:sources){
        VisNode *visNode=visNodeFact.get(v,0, sources, distanceCache);
        numSettled += visNode->settled.size();
        indexByD.insert(visNode);
        hit += visNode->unprocessed.size();
        if (visNode->settled.size()> sources.size()){
            int KKK=0;
        }
        checked += sources.size();
    }
    long maxQueue=0.0;
    for(Vertex t:dests){
        if (ddests.count(t)>0){
            if (POTENTIAL){
                for (auto it=indexByV.begin(); it!=indexByV.end();it++){
                    (*it)->update_potential(t, g, landMarks, distanceCache);
                    indexByV.modify(it, VisNode::update_minDist((*it)->get_minDistance()));
                }
            }
            // count
            bool targetFinished=false;
            VisNode* previous=NULL;
            // while the queue is not empty
            while (indexByV.size()!=0) {
                maxQueue=max(maxQueue, (long) indexByV.size());
                counter++;
                set<Vertex> settled;
                // get the top element of the
                // priority heap
                auto iter = indexByD.begin();
                VisNode *p = *iter;
                previous=p;
                double nextDist;
                if (indexByD.size()>1){
                    nextDist = (*(++iter))->get_minPotentialDistance();
                } else{
                    nextDist = p->get_minPotentialDistance();
                }
                indexByV.erase(p->v);
                settled = p->settle_distances(nextDist, distanceCache);
                numSettled +=settled.size();
                // pop the element
                // check if the element is part of
                // the goal list
                // if all goals are reached
                // check for the non visited nodes
                // which are adjacent to present node
                if (visited.count(p->v) == 0) {
                    //p->v have not been visited
                    VisNode *visNode;
                    for ( auto ve = boost::out_edges(p->v, g ); ve.first != ve.second; ++ve.first){
                        bool newvis=false;
                        Vertex neighbor=target(*ve.first,g);
                        if ((neighbor != p->v) && (visited.count(neighbor) == 0)) { //it is not in visited and check for loops
                            double l = g[*ve.first].distance;
                            auto it = indexByV.find(neighbor); //check if  the neighbor is on the heap
                            if (it == indexByV.end()) { //it is not on the heap we have to add it
                                visNode = visNodeFact.get(neighbor,t, sources,distanceCache);
                                numSettled += visNode->settled.size();
                                //                                visNode->update_potential(t,g,landMarks,distanceCache);
                                newvis=true;
                                hit += visNode->unprocessed.size();
                                checked += sources.size();
                                if (visNode->unprocessed.size()> sources.size()){
                                    int KKK=0;
                                }
                            } else {
                                visNode=*it;
                                indexByV.erase(it);
                            }
                            if (visNode->indexNodeByV.size()>0){
                                for (auto s:p->settled){ //we loop on the settled of p->v
                                    Vertex currentSource = s.first;
                                    double currentDist = s.second.distance;
                                    double d = currentDist +l; //calculate the distance up to neighbor
                                    visNode->update_distance(currentSource,d);
                                }
                                auto itt=p->indexNodeByD.begin();
                                while(itt != p->indexNodeByD.end()){
                                    Vertex currentSource = (*itt)->v;
                                    double currentDist = (*itt)->distance;
                                    if (currentDist != std::numeric_limits<double>::infinity()) {
                                        //this source have already be seen by p->v
                                        double d = currentDist+l; //calculate the distance up to neighbor
                                        visNode->update_distance(currentSource, d);
                                    } else {
                                        break; // while(itt != p->indexNodeByD.end())
                                    }
                                    itt++;
                                }// mark as visited
                            }
                            indexByD.insert(visNode);
                        }
                    }
                }
/*                if (counter>10000000) {
                    cout<<"WEIRDOS"<<endl;
                    cout<<indexByV.size()<<","<<nextDist<<","<<p->v<<","<<settled.size()<<","<<p->settled.size()<<endl;
                }
                if (counter%10000==0) {
                    cout<<"RUNNING"<<endl;
                    cout<<p->v<<","<<indexByV.size()<<","<<nextDist<<","<<settled.size()<<","<<p->settled.size()<<endl;
                }*/

                auto range=equal_range(dests.begin(),dests.end(),p->v);
                if (range.first!=range.second) {
                    // if a new dest is reached
                    int indexj=range.first-dests.begin();
                    if (p->is_fullySettled()) {
                        for (auto e:p->settled) {
                            auto range1=equal_range(sources.begin(), sources.end(), e.first);
                            int indexi=range1.first-sources.begin();
                            (*ddists)[offset+indexi][indexj] =e.second.distance;
                        }
                        destsCount++;
                        visited.insert(p->v);
                        ddests.erase(p->v);
                        if (p->v==t) {
                            targetFinished=true; //while (indexByV.size()!=0)
                        }
                        visNodeFact.yield(p);
                    } else {
                        indexByD.insert(p);
                    }
                } else {
                    if (p->is_fullySettled()) {
                        visited.insert(p->v);
                        if (p->v==t) {
                            targetFinished=true;
                        }
                        visNodeFact.yield(p);
                    } else {
                        indexByD.insert(p);
                    }
                }
                if (targetFinished)
                    break;
            }
        }
        if (ddests.size()==0) {

            for (auto ite=indexByV.begin(); ite!=indexByV.end();ite++){
                visNodeFact.yield(*ite);
            }
            indexByV.clear();
            finish= true;
            break;
        }
    }
    perf.maxQueue=maxQueue;
    perf.counter=counter;
    perf.cacheHitRate = hit*1.0/checked;
    return perf;
}


std::atomic<long> totalTime{0}, numProcessedVertex{0}, numProcessedEdge{0};

void calcCurvature(Vertex v, vector<vector<float>> &MatDist, set<Edge> &edges, vector<Vertex> sources, vector<Vertex> dests){
    for(Edge e: edges){
        Vertex neighbor=target(e,g);
        if(neighbor==v)
            cout<<"BUG"<<endl;
        vector<vector<double>> vertDist(MatDist.size(),vector<double>(out_degree(neighbor, g),std::numeric_limits<double>::infinity()));
        pair<AdjacencyIterator, AdjacencyIterator> nB = adjacent_vertices(neighbor, g);
        set<Vertex> localDests;
        int cntD=0;
        for (; nB.first != nB.second; localDests.insert(*nB.first++)){
            auto range=equal_range(dests.begin(), dests.end(),*(nB.first));
            int indexj=range.first-dests.begin();
            for(int j=0;j<MatDist.size();j++){
                vertDist[j][cntD]=MatDist[j][indexj];
            }
            cntD++;
        }
        int sA=vertDist.size(), sB=vertDist[0].size();
        double uA = (double) 1 / sA ;
        double uB = (double) 1 / sB ;
        vector<double> distributionA(sA, uA);
        vector<double> distributionB(sB, uB);
        double wd = compute_EMD(distributionA, distributionB, vertDist);
        if (isnan(wd)){
            g[e].ot= 9999.0;
        } else {
            g[e].ot=wd;
        }
        numProcessedEdge++;
    }
}

void updateDistances(Graph_t& g){
    auto es = boost::edges(g);
    double delta=0.3;
    for (auto eit = es.first; eit != es.second; ++eit) {
        double curv=g[*eit].distance-g[*eit].ot;
        g[*eit].distance += -2*curv*delta;
        g[*eit].distance=max(g[*eit].distance, 0.0);
        cout<<g[*eit].distance<<","<<curv<<endl;
        logFile1<<source(*eit,g)<<","<<target(*eit,g)<<","<<g[*eit].distance<<","<<curv<<endl;
    }
}

void process(int index) {
    Task *task;
    while (tasksToDo.size()>0) {
        tasksToDo.try_take(task);
        auto t1 = high_resolution_clock::now();
        Vertex s = task->v;
        Perf perf;
        bool done=false;
        vector<Vertex> sources(task->sources), dests(task->dests);
        set<Edge> edges(task->edges);
        vector<vector<float>> *MatDist;
        int ssize = sources.size(), dsize = dests.size();
        switch (task->type){
            case FullProcess:{
                FullTask *fullTask = (FullTask *) task;
                 MatDist= new vector<vector<float>>(ssize, vector<float>(dsize,std::numeric_limits<float>::infinity()));
                if (ssize < QUANTA) {
                    perf = multisource_uniform_cost_search_seq(MatDist, 0, sources, dests, g);
                    calcCurvature(fullTask->v, *MatDist, edges, sources, dests);
                    delete MatDist;
                    numProcessedVertex++;
                    done=true;
                } else {
                    std::atomic<int> *sharedCnt = new std::atomic<int>();
                    *sharedCnt=0;
                    int iterNum = sources.size() / QUANTA;
                    auto begin = sources.begin();
                    for(int i=0;i<sources.size();i = i+QUANTA){
                        auto first=sources.begin()+i;
                        auto end=sources.begin()+i+std::min((long)QUANTA,(long)sources.size()-i);
                        vector<Vertex> partialVect(first, end);
                        PartialTask *ptask = new PartialTask(sharedCnt, MatDist, i, s, partialVect, dests, edges);
                        tasksToDo.try_add((Task *)ptask);
                        (*sharedCnt)++;
                    }
                }
                delete fullTask;
                break;}
            case PartialProcess:
                PartialTask *partialTask = (PartialTask *) task;
                cout<<"PARTIAL TASK:"<<partialTask->v<<","<<*(partialTask->sharedCnt)<<endl;
                perf = multisource_uniform_cost_search_seq(partialTask->dists, partialTask->offset,sources,dests, g);
                (*(partialTask->sharedCnt))--;
                done=true;
                if (*(partialTask->sharedCnt) == 0) {
                    calcCurvature(partialTask->v,*(partialTask->dists),partialTask->edges, partialTask->sources, partialTask->dests);
                    delete partialTask->dists;
                    delete partialTask->sharedCnt;
                    delete partialTask;
                    numProcessedVertex++;
                }
                break;
        }
        if (done){
            auto t2 = high_resolution_clock::now();
            if (s==34)
                int KKKK=0;
            duration<double, std::micro> ms_double = t2 - t1;
            totalTime += ms_double.count();
            cout << s << "," << index << "," << numProcessedVertex << "," << numProcessedEdge << ","
                 << distanceCache.hit * 1.0 / (distanceCache.hit + distanceCache.miss) << "," << visNodeFact.number()
                 << endl;
            cout << s << "," << ms_double.count() << "," << sources.size() << "," << dests.size() << "," << perf.maxQueue
                 << "," << perf.counter << "," << perf.cacheHitRate << endl;
            logFile << s << "," << index << "," << numProcessedVertex << "," << numProcessedEdge << ","
                    << distanceCache.hit * 1.0 / (distanceCache.hit + distanceCache.miss) << "," << visNodeFact.number()
                    << "," << ms_double.count() << "," << sources.size() << "," << dests.size() << "," << perf.maxQueue
                    << "," << perf.counter << "," << perf.cacheHitRate << endl;
        }
        if (numProcessedVertex % 2000 == 0) {
            ofstream preFile;
            preFile.open("/data/Curvature/processed" + to_string(numProcessedVertex) + ".gml", ofstream::out);
            write_graphml(preFile, g, dp, true);

        }
    }
}


int main() {
    Graph_t gin;
//    readGraphMLFile(gin, "/data/Curvature/graphdumps1554598943.1554599003.graphml");
    int numIteration=10;
    string inFilename="/data/Curvature/processed.0.gml";

    readGraphMLFile(gin, inFilename);
    k_core(gin,g, 2);
    for(int index=0;index<numIteration;index++){
        distanceCache. clear();
        generateTasks1(g,tasksToDo);
        Graph_t::vertex_iterator v, vend;
        Graph_t::edge_iterator ei, ei_end;
        tie(ei, ei_end) = edges(g);
        vector<Edge> edges(ei,ei_end);
        vector<double> distances(edges.size());
        //    DistanceCache distanceCache(100000000,8);
        numProcessedVertex=0;
        numProcessedEdge=0;
        int num_core=std::thread::hardware_concurrency();
        int offset=0;
        int k=0;
        string logFilename="/data/Curvature/logFile."+to_string(index)+".log", logFilename1="/data/Curvature/dest."+to_string(index)+".log";
 //       logFile.open(logFilename.c_str(), ofstream::out);
        logFile1.open(logFilename.c_str(), ofstream::out);
//    num_core=1;
        vector<thread> threads(num_core);
        for (int i=0;i<num_core;i++){
            threads[i]=std::thread(process,i);
        }
        for (int i=0;i<num_core;i++){
            threads[i].join();
        }
        updateDistances(g);
        ofstream outFile;
        string outFilename="/data/Curvature/processed."+to_string(index+1)+".gml";
        outFile.open(outFilename.c_str(), ofstream::out);
        write_graphml(outFile, g, dp, true);
//        logFile.close();
        logFile1.close();
    }
    return 0;
}