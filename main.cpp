//
// Created by Kave Salamatian on 20/04/2021.
//

#include <cstdio>
#include <iostream>
#include <utility>
#include <thread>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphml.hpp>
#include <boost/graph/filtered_graph.hpp>
//#include <boost/graph/dijkstra_shortest_paths.hpp>

#include <boost/graph/copy.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/mem_fun.hpp>
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

struct Vertex_info_road;
struct Edge_info_road;
struct Graph_info;
//typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::undirectedS, Vertex_info, Edge_info > Graph_t;
typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::undirectedS, Vertex_info_road, Edge_info_road, Graph_info > Graph_t;
typedef boost::graph_traits < Graph_t >::vertex_iterator VertexIterator;
typedef boost::graph_traits < Graph_t >::edge_iterator EdgeIterator;
typedef boost::graph_traits < Graph_t >::adjacency_iterator AdjacencyIterator;
typedef boost::graph_traits < Graph_t >::vertex_descriptor Vertex;
typedef boost::graph_traits < Graph_t >::edge_descriptor Edge;
typedef boost::property_map < Graph_t, boost::vertex_index_t >::type IndexMap;
typedef unsigned int uint;

struct Vertex_info_road {
    float X;
    float Y;
    float x;
    float y;
    std::string name;
    float lat;
    float longi;
    string label;
    float size;
    int r,g,b;
    int degree;
    int cluster;
    double eccentricity,closnesscentrality, harmonicclosnesscentrality, betweenesscentrality;
    bool active=true;
};

struct Edge_info_road {
    float dist;
    double distance=1.0;
    double weight=1.0;
    double ot;
    double curv;
    string edgeLabel;
    double delta;
//    int r,g,b;
    bool active=true;
};

struct Graph_info {
    double avgCurv=0.0;
    double stdCurv=0.0;
};

struct Vertex_info_BGP {
    std::string country;
    std::string name;
    int prefixnum;
    int prefixall;
    int astime;
    int addall;
    int addnum;
    std::string asnumber;
    int pathnum;
//    double potential;
};

struct Edge_info_BGP {
    int weight= 1;
    int pathcount;
    int edgetime;
    //int ctime;
    int prefcount;
    double distance=1.0;
    double ot=99.0;
    bool visited=false;
    double curv=99.0;
    int addCount;
};


bool POTENTIAL=true;
double EPS= 1e-1;
int QUANTA=60;
ofstream logFile, logFile1;
boost::dynamic_properties dp;

/*
 * Read a graphml
 */
void readGraphMLFile (Graph_t& designG, std::string fileName ) {

    ifstream inFile;
/*    dp.property("asNumber", get(&Vertex_info::asnumber, designG));
    dp.property("pathNum", get(&Vertex_info::pathnum, designG));
    dp.property("Country", get(&Vertex_info::country, designG));
    dp.property("Name", get(&Vertex_info::name, designG));
    dp.property("asTime", get(&Vertex_info::astime, designG));
    dp.property("prefixNum", get(&Vertex_info::prefixnum, designG));
    dp.property("prefixAll", get(&Vertex_info::prefixall, designG));
    dp.property("addAll", get(&Vertex_info::addall, designG));
    dp.property("addNum", get(&Vertex_info::addnum, designG));
    dp.property("addCount", get(&Edge_info::addCount, designG));
    dp.property("edgeTime", get(&Edge_info::edgetime, designG));
    dp.property("weight", get(&Edge_info::weight, designG));
    dp.property("distance", get(&Edge_info::distance, designG));
    dp.property("ot", get(&Edge_info::ot, designG));
    dp.property("curv", get(&Edge_info::curv, designG));
    dp.property("pathCount", get(&Edge_info::pathcount, designG));
    dp.property("prefCount", get(&Edge_info::prefcount, designG));*/

    dp.property("label", get(&Vertex_info_road::label, designG));
    dp.property("X", get(&Vertex_info_road::X, designG));
    dp.property("Y", get(&Vertex_info_road::Y, designG));
    dp.property("meta", get(&Vertex_info_road::name, designG));
    dp.property("lat", get(&Vertex_info_road::lat, designG));
    dp.property("long", get(&Vertex_info_road::longi, designG));
    dp.property("r", get(&Vertex_info_road::r, designG));
    dp.property("g", get(&Vertex_info_road::g, designG));
    dp.property("b", get(&Vertex_info_road::b, designG));
    dp.property("x", get(&Vertex_info_road::x, designG));
    dp.property("y", get(&Vertex_info_road::y, designG));
    dp.property("size", get(&Vertex_info_road::size, designG));
    dp.property("Degré", get(&Vertex_info_road::degree, designG));
    dp.property("Modularity Class", get(&Vertex_info_road::cluster, designG));
    dp.property("Eccentricity", get(&Vertex_info_road::eccentricity, designG));
    dp.property("Closeness Centrality", get(&Vertex_info_road::closnesscentrality, designG));
    dp.property("Harmonic Closeness Centrality", get(&Vertex_info_road::harmonicclosnesscentrality, designG));
    dp.property("Betweenness Centrality", get(&Vertex_info_road::betweenesscentrality, designG));
    dp.property("dist", get(&Edge_info_road::dist, designG));
    dp.property("weight", get(&Edge_info_road::weight, designG));
    dp.property("distance", get(&Edge_info_road::distance, designG));
    dp.property("ot", get(&Edge_info_road::ot, designG));
    dp.property("curv", get(&Edge_info_road::curv, designG));
    dp.property("Edge Label", get(&Edge_info_road::edgeLabel, designG));
    map<double, double> attribute_double2double1,attribute_double2double2;
    associative_property_map<map<double, double>> avgCurv_map(attribute_double2double1);
    associative_property_map<map<double, double>> stdCurv_map(attribute_double2double2);
    dp.property("avgCurv", avgCurv_map);
    dp.property("stdCurv",stdCurv_map);



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


void k_core1(Graph_t &gin, Graph_t &gout, int k){
    Graph_t::vertex_iterator v, vend, next;
    vector<int> degrees(num_vertices(gin));

    for(boost::tie(v,vend) = vertices(gin); v != vend; ++v) {
        degrees[*v]=degree(*v,gin);
    }
    set<Vertex> sset({202,244,246,275,746,245}),sset1;
    bool changed=true;
    int count=0;
    while(changed) {
        changed = false;
        for (boost::tie(v, vend) = vertices(gin); v != vend; ++v) {
            if ((degrees[*v] < k) && (degrees[*v] > 0)) {
                changed = true;
                count++;
                for (auto ve = boost::out_edges(*v, gin); ve.first != ve.second; ++ve.first) {
                    int trg=target(*ve.first, gin),src=source(*ve.first, gin);
                    degrees[src]--;
                    degrees[trg]--;
                }
            }
        }
    }
    Vertex vertex;
    map<string,Vertex> nameMap;
    for (int i=0;i<degrees.size();i++) {
        if (degrees[i]>0) {
            vertex = add_vertex(gout);
            gout[vertex].X = gin[i].X;
            gout[vertex].Y = gin[i].Y;
            gout[vertex].name = gin[i].name;
            gout[vertex].x = gin[i].x;
            gout[vertex].y = gin[i].y;
            gout[vertex].lat = gin[i].lat;
            gout[vertex].longi = gin[i].longi;
            gout[vertex].lat = gin[i].lat;
            gout[vertex].label = gin[i].label;
            gout[vertex].size = gin[i].size;
            gout[vertex].r = gin[i].r;
            gout[vertex].g = gin[i].g;
            gout[vertex].b = gin[i].b;
            gout[vertex].degree = gin[i].degree;
            gout[vertex].cluster = gin[i].cluster;
            gout[vertex].eccentricity = gin[i].eccentricity;
            gout[vertex].closnesscentrality = gin[i].closnesscentrality;
            gout[vertex].harmonicclosnesscentrality = gin[i].harmonicclosnesscentrality;
            gout[vertex].betweenesscentrality = gin[i].betweenesscentrality;
            gout[vertex].eccentricity = gin[i].eccentricity;
            nameMap[gin[i].name]=vertex;
            if (sset.count(i)>0)
                sset1.insert(vertex);

        }
    }
    string name;
    Vertex src,trg;
    for (int i=0;i<degrees.size();i++) {
        if (degrees[i]>0) {
            src = nameMap[gin[i].name];
            if (gout[src].name=="Thessaloniki")
                int KKK=0;
            for (auto ve = boost::out_edges(i, gin); ve.first != ve.second; ++ve.first) {
                if (degrees[target(*ve.first, gin)] > 0) {
                    trg = nameMap[gin[target(*ve.first, gin)].name];
                    if (!edge(src,trg,gout).second){
                        auto ed = add_edge(src, trg, gout);
                        if (ed.second){
                            gout[ed.first].dist = gin[*ve.first].dist;
                            gout[ed.first].distance = gin[*ve.first].distance;
                            gout[ed.first].weight = gin[*ve.first].weight;
                            gout[ed.first].ot = gin[*ve.first].ot;
                            gout[ed.first].curv = gin[*ve.first].curv;
                        }
                    }
                }
            }
        }
    }
    cout<<"k-core "<<k<<" size is:"<<num_vertices(gout)<<","<<num_edges(gout)<<endl;
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
/*                    (*after)[vsource].name = (*before)[vsource1].name;
                    (*after)[vsource].country = (*before)[vsource1].country;
                    (*after)[vsource].astime = (*before)[vsource1].astime;
                    (*after)[vsource].pathnum = (*before)[vsource1].pathnum;
                    (*after)[vsource].prefixnum = (*before)[vsource1].prefixnum;
                    (*after)[vsource].prefixall = (*before)[vsource1].prefixall;
                    (*after)[vsource].asnumber = (*before)[vsource1].asnumber;*/
                    (*after)[vsource].name = (*before)[vsource1].name;
                    (*after)[vsource].X = (*before)[vsource1].X;
                    (*after)[vsource].Y = (*before)[vsource1].Y;
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
/*                        (*after)[vdest].name = (*before)[vdest1].name;
                        (*after)[vdest].country = (*before)[vdest1].country;
                        (*after)[vdest].astime = (*before)[vdest1].astime;
                        (*after)[vdest].pathnum = (*before)[vdest1].pathnum;
                        (*after)[vdest].prefixnum = (*before)[vdest1].prefixnum;
                        (*after)[vdest].prefixall = (*before)[vdest1].prefixall;
                        (*after)[vdest].asnumber = (*before)[vdest1].asnumber;*/
                        (*after)[vdest].name = (*before)[vdest1].name;
                        (*after)[vdest].X = (*before)[vdest1].X;
                        (*after)[vdest].Y = (*before)[vdest1].Y;
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
                /*(*after)[e.first].weight = (*before)[*eit].weight;
                (*after)[e.first].pathcount = (*before)[*eit].pathcount;
                (*after)[e.first].edgetime = (*before)[*eit].edgetime;
                (*after)[e.first].prefcount = (*before)[*eit].prefcount;
                (*after)[e.first].distance = (*before)[*eit].distance;*/
                (*after)[e.first].dist = (*before)[*eit].dist;
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



enum TaskType {FullProcess, PartialProcess, PartialCurvature,SuperProcess};
class Task{
public:
    TaskType type;
    Vertex v;
    vector<Vertex> sources;
    vector<Vertex> dests;
    set<Edge> edges;
    Task(TaskType type, Vertex v, vector<Vertex> sources, vector<Vertex> dests, set<Edge> edges):type(type), v(v), sources(sources),
    dests(dests), edges(edges){}
   ~Task(){
        sources.clear();
        dests.clear();
        edges.clear();
    }
};

class SuperTask: public Task{
    vector<Task> subtasks;
    SuperTask(vector<Task> &tasks, vector<Vertex> &superSources, vector <Vertex> &superDests, set<Edge> &superEdges):
    Task(SuperProcess,0,superSources, superDests, superEdges), subtasks(tasks){}
};


class FullTask: public Task{
public:
    FullTask(Vertex v, vector<Vertex> &sources, vector<Vertex> &dests, set<Edge> edges):Task(FullProcess,v,sources, dests, edges)
    {};
};

class PartialTask: public Task{
public:
    std::atomic<int> *sharedCnt;
    vector<vector<double>> *dists;
    int offset;
    PartialTask(std::atomic<int> *shared, vector<vector<double>> *dists, int offset, Vertex v, vector<Vertex> sources, vector<Vertex> dests, set<Edge> edges):
    Task(PartialProcess,v,sources, dests,edges), dists(dists), offset(offset), sharedCnt(shared){type=PartialProcess;}
};

struct TaskPaircomp {
    // Comparator function
    int operator()(Task *l, Task *r) const
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
    TaskType mode=SuperProcess;
    vector<Task> tasks;
    vector<Vertex> superSources;
    vector<Vertex> superDests;
    set<Edge> superEdges;
//    if (mode==SuperProcess){
//        superTasks
//    }

    for (tie(e, eend) = edges(g); e != eend; allEdges[*e++]=false);
    int count=0;
    for (auto ed: allEdges){
        if (g[ed.first].distance==0){
            g[ed.first].curv=g[graph_bundle].avgCurv;
            g[ed.first].ot=1;
            allEdges[ed.first]=true;
        } else {
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
                sourceNeighborsSet.insert(v);
                destNeighborsSet.insert(w);
                // we should also add all neighbors links of v
                for ( auto ve = boost::out_edges(v, g ); ve.first != ve.second; ++ve.first){
                    Graph_t::adjacency_iterator nfirst, nend;
                    tt=target(*ve.first,g);
                    ss=source(*ve.first,g);
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
                vector<Vertex> sources(sourceNeighborsSet.begin(),sourceNeighborsSet.end()),
                        dests(destNeighborsSet.begin(), destNeighborsSet.end());
                if (destNeighborsSet.size()>0){
                    FullTask *task = new FullTask(v, sources, dests, edgesToProcess);
//                if (mode==SuperProcess){
//                    if (superSources.size()  sources.size())
                    //               } else {
                    tasksToDo.try_add((Task *)task);
//                }
                }
            }
//            if (count%1000==0) {
//                cout<<count<<endl;
//            }
        }
    }
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
        long key=((long)src<<32)+dst;
//        distanceCache.insert(key,value);
        key=((long)dst<<32)+src;
//        return distanceCache.insert(key,value);
    }

    double find(const Vertex src, const Vertex dst) {
        // return the distance if he finds it and -1 if not
        ThreadSafeScalableCache<long, double>::ConstAccessor ac;
        long key=((long)src<<32)+dst;
        if (distanceCache.find(ac,key)){
            hit++;
            return *ac;
        }
        key=((long)dst<<32)+src;
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

class GraphNode{
public:
    Vertex v, source;
    double dist=std::numeric_limits<double>::infinity();
    bool settled=false;

    GraphNode(Vertex v, uint source, double dist ): v(v), source(source), dist(dist){}

    void fill(Vertex vv, uint ssource, double ddist){
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


class VisNode {
private:
//    double potential=0.0;
    InternalHeap sortedDistances;
public:
    bool isDestination=false;
    double minDist;
    map<Vertex, IntNode> settled;
    set<Vertex> unprocessed;
    typename boost::multi_index::index<InternalHeap,IntNodeByV>::type& indexNodeByV = sortedDistances.get<IntNodeByV>(); // here's where index tags (=names) come in handy
    typename boost::multi_index::index<InternalHeap,IntNodeByD>::type& indexNodeByD = sortedDistances.get<IntNodeByD>();
    Vertex v;

    VisNode(Vertex v, vector<Vertex> &sources, set<Vertex> & dests): v(v){
        double dist=-1.0, mdist=std::numeric_limits<double>::infinity();
        for(Vertex w:sources){
            if (v==w){
                indexNodeByD.insert(new IntNode(w,0));
//                settled[w].distance=0;
//                settled[w].v=w;
                unprocessed.insert(w);
                mdist=0;
            } else {
//                dist=distanceCache.find(v,w);
//                if  (dist>-1){
//                    //indexNodeByD.insert(new IntNode(w,dist));
//                    settled[w].distance=dist;
//                    settled[w].v=w;
//                    settled[w].visited=false;
//                    unprocessed.insert(w);
//                    mdist=min(mdist,dist);
//                } else {
                    indexNodeByD.insert(new IntNode(w,std::numeric_limits<double>::infinity()));
//                }
            }
        }
/*        if (POTENTIAL){
            if (dests.count(v)>0){
                potential=EPS;
            } else{
                potential=0;
            }*/
        if (dests.count(v)>0) {
            isDestination = true;
        }

//            dist=distanceCache.find(v,target);
//            if (dist>-1){
//                potential=dist;
//            } else {
//            }
//        }
        minDist=mdist;
    }

    ~VisNode(){
        for (auto p:indexNodeByV){
            delete p;
        }
        indexNodeByV.clear();
        sortedDistances.clear();
        settled.clear();
        unprocessed.clear();
    }

    void empty(){
        for (auto p:indexNodeByV){
            delete p;
        }
        indexNodeByV.clear();
        sortedDistances.clear();
        settled.clear();
        unprocessed.clear();
    }

    void fill(Vertex w, vector<Vertex> &ssources,  set<Vertex> &ddests ){
        v=w;
        double dist=-1.0, mdist=std::numeric_limits<double>::infinity();
        for(Vertex w:ssources){
            if (v==w){
                indexNodeByD.insert(new IntNode(w,0));
//                settled[w].distance=0;
//                settled[w].v=w;
                unprocessed.insert(w);
                mdist=0;
            } else {
//                dist=ddistanceCache.find(v,w);
//               if  (dist>-1) {
//                    indexNodeByD.insert(new IntNode(w,dist));
//                    settled[w].distance=dist;
//                    settled[w].v=w;
//                    settled[w].visited=false;
//                    unprocessed.insert(w);
//                    mdist=min(mdist,dist);
//                } else {
                    indexNodeByD.insert(new IntNode(w,std::numeric_limits<double>::infinity()));
//                }
            }
        }
/*        if (POTENTIAL){
            if (ddests.count(v)>0){
                potential=EPS;
            } else{
                potential=0;
            }*/
        if (ddests.count(v)>0) {
            isDestination = true;
        }
//            dist=ddistanceCache.find(v,target);
//            if (dist>-1){
//                potential=dist;
//            } else {
//            }
//        }
        minDist=mdist;
    }

    double get_potential(Vertex node,Graph_t &g, set<Vertex> &landMarks, Vertex target, DistanceCache &distanceCache){
        double maxPotential=0.0;
        for(Vertex v:landMarks) {
            double dnode=1.0, dtarget=1.0;
            dnode=distanceCache.find(v,node);
            dtarget=distanceCache.find(v,target);

            if ((dnode>-1) && (dtarget>-1))
                maxPotential=max(maxPotential,abs(dnode-dtarget));
        }
        return maxPotential;
    }


    IntNode get_settledIntNode(Vertex w){
        return settled[w];
    }

    bool update_distance(Vertex w, double d){
//        if (settled.count(w)==0){
            auto it=indexNodeByV.find(w);
            if ((*it)->distance>d){
                indexNodeByV.modify(it,IntNode::distChange(d));
                double minDistC=minDist;
                if (minDist>d)
                    minDist=d;
                if (minDist<minDistC)
                    return true;
            }
 //       }
        return false;
    }

/*    void update_potential(Vertex target, Graph_t &g, set<Vertex> landMarks, DistanceCache &distanceCache){
        double dist = 1.0;
        dist=distanceCache.find(v,target);
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
    }*/

    double get_minDistance(){
        if (indexNodeByV.size()>0)
            return minDist;
//            return (*(indexNodeByD.begin()))->distance;
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
        void operator()(VisNode *node) { node->minDist = d;}
    };

    bool settle_distance(Vertex src) {
        auto itt=indexNodeByV.find(src);
        if (itt != indexNodeByV.end()) {
            settled[src] = *(*itt);
            delete *itt;
            itt = indexNodeByV.erase(itt);
            if (!is_fullySettled()) {
                minDist=(*indexNodeByD.begin())->distance;
            } else {
                minDist=std::numeric_limits<double>::infinity();
            }
            return true;
        }
        return false;
    }


    set<Vertex> settle_distances(double nextDist) {// nextDist is the smallest distance in the queue
        //check if any distance can be settled
        set<Vertex> newSettled;
        auto it =indexNodeByD.begin();
//        for(Vertex u:unprocessed){// value that are settled but not enough put in settled vector
//            newSettled.insert(u);
//        }
        unprocessed.clear();
        while (it != indexNodeByD.end()) {
            Vertex currentSource = (*it)->v;
            double currentDistance = (*it)->distance;
//            if (currentDistance !=
//            std::numeric_limits<double>::infinity()) {// node v has already been reached from currentSource
                if (currentDistance <= nextDist) {
                    newSettled.insert(currentSource);
                    settled[currentSource] = *(*it);
//                    distanceCache.insert(currentSource, v,currentDistance);
                    delete *it;
                    it = indexNodeByD.erase(it);
                } else {
//                    it++;
                    break;
                }
            } //else{
//                break;
//            }
//        }
        if (!is_fullySettled()) {
            minDist=(*indexNodeByD.begin())->distance;
        } else {
            minDist=std::numeric_limits<double>::infinity();
        }
        return newSettled;
    }

    bool is_fullySettled(){
        return (indexNodeByV.size()==0);
    }

    size_t getSettledSize() const {
        return (settled.size());
    }
};

struct Comp {
    bool operator()(const size_t &a, const size_t &b) const {
        return a > b;
    }
};

using Heap = multi_index_container< VisNode*, // the data type stored
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
                boost::multi_index::member<VisNode, bool, &VisNode::isDestination>,
                boost::multi_index::const_mem_fun<VisNode,std::size_t, &VisNode::getSettledSize>
            >
        >
    >
>;


class GraphNodeArray{
public:
    int graphSize;
    vector<Vertex> &sources;
    double *nodeArray;
//    double *minDist;
    int *settledCount;
    uint *minSource;
    GraphNodeArray(int graphSize, vector<Vertex> &sources): graphSize(graphSize), sources(sources) {
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
        if ((vind==163) && (sourceind==2))
            int KKKK=0;
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
            graphNodeQueue.try_pop(elem);
            delete elem;
        }
    }

    GraphNode *get(Vertex v, uint source, double dist){
        GraphNode *elem;
        if ((v==163) && (source==2))
            int KKKK=0;
        int len= graphNodeQueue.unsafe_size();
        if (!graphNodeQueue.try_pop(elem)){
            elem=new GraphNode(v,source,dist);
            numb++;
        } else {
            elem->  fill(v,source, dist);
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
    tbb::concurrent_queue<GraphNode*> graphNodeQueue;
    std::atomic<long> numb{0}, yi{0};

};

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

    VisNode *get(Vertex v, vector<Vertex> &sources, set<Vertex> &dests){
        VisNode *elem;
        if (!visNodeQueue.try_pop(elem)){
            elem=new VisNode(v, sources, dests);
            numb++;
        } else {
            elem->fill(v, sources, dests);
        }
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
//DistanceCache distanceCache(2000000,48);
//DistanceCache distanceCache1(2000000,48);
VisNodeFactory visNodeFact;
GraphNodeFactory graphNodeFact;

struct myComp {
    constexpr bool operator()(
            GraphNode* const& a,
            GraphNode* const& b)
    const noexcept
    {
        return a->dist > b->dist;
    }
};


//#define EPS 0.0
#define ALPHA 0.5
Perf multisource_uniform_cost_search_seq1(vector<vector<double>> *ddists, int offset, vector<Vertex> &sources, vector<Vertex> &dests, Graph_t &g){
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

    GraphNodeArray graphNodeArray(num_vertices(g),sources);
    priority_queue<GraphNode*, vector<GraphNode*>, myComp > *costQueue, *bakCostQueue, *swap;
    costQueue=new priority_queue<GraphNode*, vector<GraphNode*>, myComp >();
    std::set<Vertex> seen;

    int counter=0;
    int numSettled=0;
    int destsCount=0;
    bool finish=false;
    set<Vertex> ddests(dests.begin(), dests.end());
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





Perf multisource_uniform_cost_search_seq(vector<vector<double>> *ddists, int offset, vector<Vertex> &sources, vector<Vertex> &dests, Graph_t &g){
    // minimum cost upto
    // goal state from starting
    // state

    // insert the starting index
    // map to store visited node
    set<Vertex> landMarks;
    Perf perf;
    perf.sourceSize=sources.size();
    perf.destSize=dests.size();
    map<Vertex,int> srcCount;
    for (Vertex src:sources)
        srcCount[src]=0;
    long hit=0, checked=0;
    set<Vertex> visited;
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
        VisNode *visNode=visNodeFact.get(v, sources, ddests);
        numSettled += visNode->settled.size();
        indexByD.insert(visNode);
        hit += visNode->unprocessed.size();
        checked += sources.size();
    }
    long maxQueue=0.0;
    for(Vertex t:dests) {
        if (ddests.count(t) > 0) {
//            if (POTENTIAL){
//                for (auto it=indexByV.begin(); it!=indexByV.end();it++){
//                    (*it)->update_potential(t, g, landMarks, distanceCache);
//                    indexByV.modify(it, VisNode::update_minDist((*it)->get_minDistance()));
//                }
        }
        // count
        bool targetFinished = false;
        VisNode *previous = NULL;
        // while the queue is not empty
        while (indexByV.size() != 0) {
            maxQueue = max(maxQueue, (long) indexByV.size());
            counter++;
            set<Vertex> settled;
            // get the top element of the
            // priority heap
            auto iter = indexByD.begin(); //get the smallest of the queue
            VisNode *p = *iter;
            previous = p;
            indexByV.erase(p->v); // erase from the queue
            // pop the element
            // check if the element is part of
            // the goal list
            // if all goals are reached
            // check for the non visited nodes
            // which are adjacent to present node
            if (visited.count(p->v) == 0) {
                //p->v have not been visited
                VisNode *visNode;
                for (auto ve = boost::out_edges(p->v, g); ve.first != ve.second; ++ve.first) {
                    //loop on neighbors
                    Vertex neighbor = target(*ve.first, g);
                    if ((neighbor != p->v) && (visited.count(neighbor) == 0)) {
                        //if it is not fully settled (and in visited) and check for loops
                        double l = g[*ve.first].distance;
                        auto it = indexByV.find(neighbor); //check if  the neighbor is on the heap
                        if (it == indexByV.end()) { //it is not on the heap we have to add it
                            visNode = visNodeFact.get(neighbor, sources, ddests);
                            numSettled += visNode->settled.size();
                            //                                visNode->update_potential(t,g,landMarks,distanceCache);
                            hit += visNode->unprocessed.size();
                            checked += sources.size();
                            if (visNode->unprocessed.size() > sources.size()) {
                                int KKK = 0;
                            }
                        } else {
                            visNode = *it;
                            indexByV.erase(it);
                        }
                        // visNode is the neighbor that is going to be updated
                        // we already know that visNode is not fully settled !
/*  THIS LOOP IS NOT ANYMORE NEEDED AS A SETTLED VALUE HAS ALREADY GONE TROUGH UPDATING ITS NEIGHBORS
 *                          for (auto s:p->settled){ //we loop on the settled of p->v

                                Vertex currentSource = s.first;
                                double currentDist = s.second.distance;
                                double d = currentDist +l; //calculate the distance up to neighbor
                                visNode->update_distance(currentSource,d);
                            }*/

                        auto itt = p->indexNodeByD.begin();
                        while (itt != p->indexNodeByD.end()) {
                            Vertex currentSource = (*itt)->v;
                            double currentDist = (*itt)->distance;
                            if (visNode->settled.count(currentSource) == 0) {
                                if (currentDist != std::numeric_limits<double>::infinity()) {
                                    //this source have already be seen by p->v
                                    double d = currentDist + l; //calculate the distance up to neighbor
                                    visNode->update_distance(currentSource, d);
                                } else {
                                    break; // while(itt != p->indexNodeByD.end())
                                }
                            }
                            itt++;
                        }// mark as visited
                        indexByD.insert(visNode);
                    }
                }
                double nextDist; // the penultimate distance in the queue
                iter = indexByD.begin(); //get the smallest of the queue
                if (indexByD.size()>1){
                    nextDist = (*(++iter))->get_minPotentialDistance();
                } else{
                    nextDist = std::numeric_limits<double>::infinity();
                }
                settled = p->settle_distances(nextDist);
                if (ddests.count(p->v)>0) {
                    set<Vertex> finishedSrc;
                    for (Vertex v: settled) {
                        srcCount[v]++;
                        if (srcCount[v] >= dests.size()) {
                            finishedSrc.insert(v);
                        }
                    }
                    if (finishedSrc.size() > 0) {
                         for (auto it = indexByV.begin(); it != indexByV.end(); it++) {
                            VisNode *pp = *it;
                             if (pp != p) {
                                 for (Vertex v: finishedSrc) {
                                     pp->settle_distance(v);
                                 }
                                 if (pp->is_fullySettled()) {
                                     if (ddests.erase(pp->v) > 0) { //pp is a destination
                                         destsCount++;
                                         if (pp->v == t) {
                                             targetFinished = true; //while (indexByV.size()!=0)
                                         }
                                         auto range = equal_range(dests.begin(), dests.end(), pp->v);
                                         int indexj = range.first - dests.begin();
                                         for (auto e: pp->settled) {
                                             auto range1 = equal_range(sources.begin(), sources.end(), e.first);
                                             int indexi = range1.first - sources.begin();
                                             (*ddists)[offset + indexi][indexj] = e.second.distance;
                                         }
                                     }
                                     indexByV.erase(pp->v);
                                     visited.insert(pp->v);
                                     visNodeFact.yield(pp);
                                 }
                             }//check if it has not been already settled
                        }
                    }
                    numSettled += settled.size();
                }
            }
            if (p->is_fullySettled()) {
                if (ddests.erase(p->v) > 0) { //p is a destination
                    destsCount++;
                    if (p->v == t) {
                        targetFinished = true; //while (indexByV.size()!=0)
                    }
                    auto range = equal_range(dests.begin(), dests.end(), p->v);
                    int indexj = range.first - dests.begin();
                    for (auto e: p->settled) {
                        auto range1 = equal_range(sources.begin(), sources.end(), e.first);
                        int indexi = range1.first - sources.begin();
                        (*ddists)[offset + indexi][indexj] = e.second.distance;
                    }
                }
                visNodeFact.yield(p);
                visited.insert(p->v);
            } else {
                indexByD.insert(p);
            }
            if (targetFinished) {
                break;
            }
        }
        if (ddests.size() == 0) {
             for (auto ite = indexByV.begin(); ite != indexByV.end(); ite++) {
                visNodeFact.yield(*ite);
            }
            indexByV.clear();
            finish = true;
            break;
        }
    }
    perf.maxQueue=maxQueue;
    perf.counter=counter;
    perf.cacheHitRate = hit*1.0/checked;
    return perf;
}


std::atomic<long>  numProcessedVertex{0}, numProcessedEdge{0};

void calcCurvature(double alpha, Vertex v, vector<vector<double>> &MatDist, set<Edge> &edges, vector<Vertex> sources, vector<Vertex> dests, Graph_t &g){
    vector<double> distributionA;
    vector<double> distributionB;
    double wd;
    for(Edge e: edges){
        Vertex src=source(e,g);
        Vertex neighbor=target(e,g);
        if ((g[src].name=="Lyon") && (g[neighbor].name=="Chambéry")){
            int  KKKK=0 ;
        }
        vector<vector<double>> vertDist(MatDist.size(),vector<double>(out_degree(neighbor, g),std::numeric_limits<double>::infinity()));
        pair<AdjacencyIterator, AdjacencyIterator> nB = adjacent_vertices(neighbor, g);
        set<Vertex> localDests;
        int cntD=0, destIndex=0;
        for (; nB.first != nB.second; localDests.insert(*nB.first++)){
            auto range=equal_range(dests.begin(), dests.end(),*(nB.first));
            int indexj=range.first-dests.begin();
            for(int j=0;j<MatDist.size();j++){
                vertDist[j][cntD]=MatDist[j][indexj];
            }
            if (*(nB.first)==neighbor)
                destIndex=cntD;
            cntD++;
        }
        int sA=vertDist.size(), sB=vertDist[0].size();
        if (sB>1){
            double uA = (double) (1-alpha) / (sA-1) ;
            double uB = (double) (1-alpha) / (sB-1) ;
            distributionA.resize(sA);
            fill_n(distributionA.begin(),sA,uA);
            distributionB.resize(sB);
            fill_n(distributionB.begin(),sB,uB);
//        vector<double> distributionA(sA, uA);
//        vector<double> distributionB(sB, uB);
            auto range=equal_range(sources.begin(), sources.end(),src);
            int index=range.first-sources.begin();
            distributionA[index]=alpha;
            distributionB[destIndex]=alpha;

            wd = compute_EMD(distributionA, distributionB, vertDist);
            if (isnan(wd)){
                g[e].ot= 9999.0;
                g[e].curv=9999.0;
            } else {
                g[e].ot=wd;
                if (g[e].distance<EPS){
                    g[e].curv=g[graph_bundle].avgCurv;
                } else {
                    g[e].curv=1-wd/g[e].distance;
                    if (g[e].curv<-3){
                        int KKKK=0;
                    }
                }
            }
            distributionA.clear();
            distributionB.clear();
        } else {
            numProcessedEdge++;
            g[e].curv=0.0;
            g[e].ot=0.0;
            g[e].distance=EPS/2;
        }
        numProcessedEdge++;
    }
}

bool updateDistances(Graph_t &g){

    auto es = edges(g);
    double delta=0.215;
    double sumWeights=0.0;
    int numEdgesUnfiltered=0;
    bool surgery=false;
    for (auto eit = es.first; eit != es.second; ++eit) {
//        double curv=1-g[*eit].ot/g[*eit].distance;
        double ddd=g[*eit].distance;
        g[*eit].distance -=2*g[*eit].curv*delta;
        if (g[*eit].distance<EPS){ //we need a surgery of type 1
            surgery=true;
            Vertex src=source(*eit,g), dst=target(*eit,g);
            cout<<"Surgery Type 1: "<<src<<":"<<dst<<","<<g[src].name<<":"<<g[dst].name<<", Curvature:"<<g[*eit].curv<<endl;
            g[src].name=g[src].name+","+g[dst].name;
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
                        add_edge(src, ddst, {g[*ve.first].dist, g[*ve.first].distance, g[*ve.first].weight, g[*ve.first].ot,
                                             g[*ve.first].curv, g[*ve.first].edgeLabel, g[*ve.first].delta,
                                             g[*ve.first].active}, g);
                        g[*ve.first].active = false;
                    }
                    if (g[*ve.first].distance<0)
                        int KKKK=0;
                }
            }
        } else {
            sumWeights +=g[*eit].distance;
            numEdgesUnfiltered++;
        }
    }
    int numV=num_vertices(g);
    int numE= num_edges(g);
    double rescaling=numEdgesUnfiltered*1.0/sumWeights;
    double maxdist=-1.0, mindist=999999.0;
    double sumCurv=0.0, sum2Curv=0.0;
    Edge maxEdge,minEdge;
    es = edges(g);
    for (auto eit = es.first; eit != es.second; ++eit) {
        if (g[*eit].active){
            double ddd=g[*eit].distance;
            Vertex src=source(*eit,g), dst=target(*eit,g);
            g[*eit].distance =g[*eit].distance*rescaling;
            if ((g[src].name=="Lyon") && (g[dst].name=="Chambéry")){
                int  KKKK=0 ;
                cout<<"curv:"<<g[*eit].curv<<" dist:"<<g[*eit].distance<<endl;
            }
            if (g[*eit].distance>maxdist) {
                maxdist=g[*eit].distance;
                maxEdge=*eit;
            }
            if (g[*eit].distance<mindist) {
                mindist=g[*eit].distance;
            }
            sumCurv += g[*eit].curv;
            sum2Curv += g[*eit].curv*g[*eit].curv;
//        cout<<"res "<<g[source(*eit,g)].name<<","<<g[target(*eit,g)].name<<","<<g[*eit].distance<<","<<g[*eit].curv<<endl;
            logFile1<<g[source(*eit,g)].name<<","<<g[target(*eit,g)].name<<","<<source(*eit,g)<<","<<target(*eit,g)<<","<<
                    g[*eit].distance<<","<<g[*eit].curv<<","<<g[*eit].ot<<endl;

        }
    }
    g[graph_bundle].avgCurv=sumCurv/numEdgesUnfiltered;
    g[graph_bundle].stdCurv=sqrt(sum2Curv/numEdgesUnfiltered-g[graph_bundle].avgCurv*g[graph_bundle].avgCurv);
    cout <<"maxdist:"<<maxdist<<",mindist:"<<mindist<<", maxEdge "<<g[source(maxEdge,g)].name<<":"<<g[target(maxEdge,g)].name
    <<" curv "<<g[maxEdge].curv<<endl;
    cout<<"avgCurv="<<g[graph_bundle].avgCurv<<", stdCurv="<<g[graph_bundle].stdCurv<<endl;
    return surgery;
}

void process(int threadIndex, Graph_t *g) {
    Task *task;
    long totalTime1=0, totalTime2=0;
    while (tasksToDo.try_take(task)== BlockingCollectionStatus::Ok) {
//        tasksToDo.try_take(task);
        auto t1 = high_resolution_clock::now(), t2=t1,t3=t1;
        duration<double, std::micro> ms_double1, ms_double2;
        Vertex s = task->v;
        Perf perf1, perf2;
        bool done=false;
        vector<Vertex> sources(task->sources), dests(task->dests);
        set<Edge> edges(task->edges);
        vector<vector<double>> *MatDist1, *MatDist2;
//        vector<vector<double>> *MatDist;
        int ssize = sources.size(), dsize = dests.size();
        switch (task->type){
            case FullProcess:{
                FullTask *fullTask = (FullTask *) task;
                MatDist1= new vector<vector<double>>(ssize, vector<double>(dsize,std::numeric_limits<double>::infinity()));
                MatDist2= new vector<vector<double>>(ssize, vector<double>(dsize,std::numeric_limits<double>::infinity()));
                if (ssize < QUANTA) {
                    perf1 = multisource_uniform_cost_search_seq1(MatDist1, 0, sources, dests, *g);
                    t2 = high_resolution_clock::now();
//                    perf2 = multisource_uniform_cost_search_seq(MatDist2, 0, sources, dests, *g);
//check for InF
                    for (int i=0;i<ssize;i++){
                        for (int j=0;j<dsize;j++) {
                            if (isinf((*MatDist1)[i][j])) {
//                                multisource_uniform_cost_search_seq(MatDist2, 0, sources, dests, *g);
                                int KKKK=0;
                            }
//                            if ((*MatDist1)[i][j] != (*MatDist2)[i][j])
//                                cout << "NOT EQUAL  " << i << "," <<j << ","<<(*MatDist1)[i][j]<< ","<<(*MatDist2)[i][j]
//                                <<","<<(*MatDist1)[i][j]-(*MatDist2)[i][j]<<endl;
                         }
                    }

                    t3 = high_resolution_clock::now()  ;
//                    t2 = high_resolution_clock::now();
//                    for(int i=0; i<ssize; i++){
//                        for (int j; j<dsize; j++){
//                            if ((*MatDist1)[i][j]!=(*MatDist2)[i][j])
//                                cout<<"NOT EQUAL "<<i<<","<<j<<endl;
//                        }
//                    }
//                    if ((*MatDist1)[0][0]==std::numeric_limits<double>::infinity())
//                        int KKKK=0;
                    calcCurvature(ALPHA, fullTask->v, *MatDist1, edges, sources, dests,*g);
                    delete MatDist1;
                    delete MatDist2;
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
                        PartialTask *ptask = new PartialTask(sharedCnt, MatDist2, i, s, partialVect, dests, edges);
                        tasksToDo.try_add((Task *)ptask);
                        (*sharedCnt)++;
                    }
                }
                delete fullTask;
                break;}
            case PartialProcess:
                PartialTask *partialTask = (PartialTask *) task;
                cout<<"PARTIAL TASK:"<<partialTask->v<<","<<*(partialTask->sharedCnt)<<endl;
                t2 = high_resolution_clock::now();
                perf2 = multisource_uniform_cost_search_seq(partialTask->dists, partialTask->offset,sources,dests, *g);
                t3 = high_resolution_clock::now();
                (*(partialTask->sharedCnt))--;
                done=true;
                if (*(partialTask->sharedCnt) == 0) {
                    calcCurvature(ALPHA, partialTask->v,*(partialTask->dists),partialTask->edges, partialTask->sources, partialTask->dests,*g);
                    delete partialTask->dists;
                    delete partialTask->sharedCnt;
                    numProcessedVertex++;
                }
                delete partialTask;
                break;
        }
        if (done){
//            ms_double1 = t2 - t1;
            ms_double2 = t3 - t2;
//            totalTime1 += ms_double1.count();
            totalTime2 += ms_double2.count();
//            cout << s << "," << threadIndex << "," << numProcessedVertex << "," << numProcessedEdge <<","<<visNodeFact.number()
//                 << endl;
//            cout << s << "," <<totalTime1<<"," << ms_double1.count() << "," << sources.size() << "," << dests.size() <<
//                "," << perf1.maxQueue << "," << perf1.counter << "," << perf1.cacheHitRate << endl;
//            cout << s << "," <<totalTime2<< "," << ms_double2.count() << "," << sources.size() << "," << dests.size() <<
//                "," << perf2.maxQueue << "," << perf2.counter<<"," <<ms_double2.count()*1.0/(sources.size()*dests.size())<< endl;
            logFile << s << "," << threadIndex << "," << numProcessedVertex << "," << numProcessedEdge << ","
                    << visNodeFact.number() << "," << ms_double2.count() << "," << sources.size() << "," << dests.size() << ","
                    << perf2.maxQueue << "," << perf2.counter << "," << ms_double2.count()*1.0/(sources.size()*dests.size())<<endl;
        }
        if (numProcessedVertex % 2000 == 0) {
            ofstream preFile;
            preFile.open("/data/Curvature/processed" + to_string(numProcessedVertex) + ".graphml", ofstream::out);
            boost::dynamic_properties dpout;
            write_graphml(preFile, *g, dpout, true);
        }
    }
}


int main(int argc, char **argv)  {
    Graph_t *g=new Graph_t, *gin=new Graph_t, *ginter;

    struct Predicate {// both edge and vertex
//    Predicate(Graph_t *g): g(g){};
//    Predicate(){};
        bool operator()(Edge e) const      {return (*g)[e].active;}
        bool operator()(Vertex vd) const { return (*g)[vd].active; }
        Graph_t *g;
    };


    using Filtered_Graph_t = boost::filtered_graph<Graph_t, Predicate, Predicate>;


    string filename="graphdumps1554598943.1554599003.graphml",path="/data/Curvature/";
    int iterationIndex=0;
    if( argc > 2 ) {
        string command1(argv[1]);
        if (command1 == "-P") {
            path=string(argv[2]);
        }
        string command2(argv[3]);
        if (command2 =="-F")
            filename= string(argv[4]);
        string command3(argv[5]);
        if (command3 =="-I")
            iterationIndex= stoi(argv[6]);

    }

//    Graph_t gin;
//    string inFilename = "/Users/ksalamatian/graphdumps1554598943.1554599003.graphml";
    readGraphMLFile(*gin, path+"/"+filename);
    int numIteration=2000;
//    string inFilename="/data/Curvature/processed.0.gml";

//    readGraphMLFile(gin, inFilename);

    k_core1(*gin,*g, 2);

    for(int index=iterationIndex;index<iterationIndex+numIteration;index++){
//        distanceCache. clear();
        cout<<"Index:"<<index<<" ";
        generateTasks1(*g,tasksToDo);
        Graph_t::vertex_iterator v, vend;
        Graph_t::edge_iterator ei, ei_end;
        tie(ei, ei_end) = edges(*g);
        vector<Edge> edges(ei,ei_end);
        vector<double> distances(edges.size());
        //    DistanceCache distanceCache(100000000,8);
        numProcessedVertex=0;
        numProcessedEdge=0;
        int num_core=std::thread::hardware_concurrency();
//        num_core=1;
        int offset=0;
        int k=0;
        string logFilename=path+"/logFile."+to_string(index)+".log", logFilename1=path+"/dest."+to_string(index)+".log";
        logFile.open(logFilename.c_str(), ofstream::out);
        logFile1.open(logFilename1.c_str(), ofstream::out);
//    num_core=1;
        vector<thread> threads(num_core);
        for (int i=0;i<num_core;i++){
            threads[i]=std::thread(process,i,g);
        }
        for (int i=0;i<num_core;i++){
            threads[i].join();
        }
        if (updateDistances(*g)){
            Predicate predicate{g};
            ginter = new Graph_t();
            Filtered_Graph_t fg(*g, predicate, predicate);
            copy_graph(fg,*ginter);
            g->clear();
            delete g;
            g=ginter;
        }
        ofstream outFile;
        string outFilename=path+"/processed."+to_string(index+1)+".graphml";
        outFile.open(outFilename.c_str(), ofstream::out);
//        for(boost::tie(v,vend) = vertices(*g); v != vend; ++v) {
//            cout<<(*g)[*v].name<<endl;
//        }
        boost::dynamic_properties dpout;

        dpout.property("label", get(&Vertex_info_road::label, *g));
        dpout.property("X", get(&Vertex_info_road::X, *g));
        dpout.property("Y", get(&Vertex_info_road::Y, *g));
        dpout.property("meta", get(&Vertex_info_road::name, *g));
        dpout.property("lat", get(&Vertex_info_road::lat, *g));
        dpout.property("long", get(&Vertex_info_road::longi, *g));
        dpout.property("r", get(&Vertex_info_road::r, *g));
        dpout.property("g", get(&Vertex_info_road::g, *g));
        dpout.property("b", get(&Vertex_info_road::b, *g));
        dpout.property("x", get(&Vertex_info_road::x, *g));
        dpout.property("y", get(&Vertex_info_road::y, *g));
        dpout.property("size", get(&Vertex_info_road::size, *g));
        dpout.property("Degré", get(&Vertex_info_road::degree, *g));
        dpout.property("Modularity Class", get(&Vertex_info_road::cluster, *g));
        dpout.property("Eccentricity", get(&Vertex_info_road::eccentricity, *g));
        dpout.property("Closeness Centrality", get(&Vertex_info_road::closnesscentrality, *g));
        dpout.property("Harmonic Closeness Centrality", get(&Vertex_info_road::harmonicclosnesscentrality, *g));
        dpout.property("Betweenness Centrality", get(&Vertex_info_road::betweenesscentrality, *g));


        dpout.property("dist", get(&Edge_info_road::dist, *g));
        dpout.property("weight", get(&Edge_info_road::weight, *g));
        dpout.property("distance", get(&Edge_info_road::distance, *g));
        dpout.property("ot", get(&Edge_info_road::ot, *g));
        dpout.property("curv", get(&Edge_info_road::curv, *g));

        map<double, double> attribute_double2double1,attribute_double2double2;
        associative_property_map<map<double, double>> avgCurv_map(attribute_double2double1);
        associative_property_map<map<double, double>> stdCurv_map(attribute_double2double2);
        dpout.property("avgCurv", avgCurv_map);
        dpout.property("stdCurv",stdCurv_map);

          write_graphml(outFile, *g, dpout, true);
//        logFile.close();
        logFile1.close();
    }
    return 0;
}