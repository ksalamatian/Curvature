#include <stdio.h>
#include <iostream>
#include<utility>
#include <string>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphml.hpp>
#include <boost/property_map/dynamic_property_map.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/copy.hpp>
#include <fstream>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include "emd.h"
#include <vector>
#include <set>
#include <tbb/parallel_for.h>
#include<chrono>
#include <queue>



using namespace boost;
using namespace std;
using namespace tbb;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

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
};
struct Edge_info {
    int weight= 1;
    long pathcount;
    int edgetime;
    int prefcount;
    double distance=1.0;
    double curvature=0.0;
};
typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::undirectedS, Vertex_info, Edge_info > Graph_t;
typedef boost::graph_traits < Graph_t >::vertex_iterator VertexIterator;
typedef boost::graph_traits < Graph_t >::edge_iterator EdgeIterator;
typedef boost::graph_traits < Graph_t >::adjacency_iterator AdjacencyIterator;
typedef boost::graph_traits < Graph_t >::vertex_descriptor Vertex;
typedef boost::graph_traits < Graph_t >::edge_descriptor Edge;
typedef boost::property_map < Graph_t, boost::vertex_index_t >::type IndexMap;


/*
 * Read a graphml
 */
void readGraphMLFile ( Graph_t& designG, std::string fileName ) {
    boost::dynamic_properties dp;
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

    ifstream inFile;
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

void k_core(Graph_t gin, Graph_t &gout, int k){

    Graph_t *before, *after, empty;
    before= &gin;
    Graph_t::vertex_descriptor vsource, vsource1,vdest, vdest1;
    std::pair<Graph_t::edge_descriptor, bool> e;
    cout<<"k-core 0 size is:"<<num_vertices(gin)<<","<<num_edges(gin)<<endl;
    for(int i=0;i<k;i++) {
        Graph_t *after, empty;
        after=&empty;
        vector<int> color(num_vertices(*before), 0);
        IndexMap index = get(vertex_index, *before);
        bool addEdge, sourceOK, destOK;
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

vector<double> uniform_cost_search(vector<int> goal, int start, vector<vector<int>> g1, map<pair<int, int>, double> cost)
{
    // minimum cost upto
    // goal state from starting
    // state
    vector<double> answer;

    // create a priority queue
    priority_queue<pair<int, int> > queue;

    // set the answer vector to max value
    for (int i = 0; i < goal.size(); i++)
        answer.push_back(FLT_MAX);

    // insert the starting index
    queue.push(make_pair(0, start));

    // map to store visited node
    map<int, int> visited;

    // count
    int count = 0;

    // while the queue is not empty
    while (queue.size() > 0) {

        // get the top element of the
        // priority queue
        pair<int, int> p = queue.top();

        // pop the element
        queue.pop();

        // get the original value
        p.first *= -1;

        // check if the element is part of
        // the goal list
        if (find(goal.begin(), goal.end(), p.second) != goal.end()) {

            // get the position
            int index = find(goal.begin(), goal.end(),
                             p.second) - goal.begin();

            // if a new goal is reached
            if (answer[index] == INT_MAX)
                count++;

            // if the cost is less
            if (answer[index] > p.first)
                answer[index] = p.first;

            // pop the element
            queue.pop();

            // if all goals are reached
            if (count == goal.size())
                return answer;
        }

        // check for the non visited nodes
        // which are adjacent to present node
        if (visited[p.second] == 0)
            for (int i = 0; i < g1[p.second].size(); i++) {

                // value is multiplied by -1 so that
                // least priority is at the top
                queue.push(make_pair((p.first +
                                      cost[make_pair(p.second, g1[p.second][i])]) * -1,
                                     g1[p.second][i]));
            }

        // mark as visited
        visited[p.second] = 1;
    }

    return answer;
}


/*
 * Return the distance matrix between the edge' source and target neighbors
 */
vector<vector<double>> calcul_chemin(int s, int d, vector<vector<int>> g, map<pair<int, int>, double> cost) {

    int lS = g[s].size();
    int lD = g[d].size();


    vector<vector<double>> MatDistance(lS,vector<double>(lD,0.0));

    int i=0;
    for (auto ii : g[s]) {
        auto t1 = high_resolution_clock::now();

        vector<int> goals = g[d];
        vector<double> distgoals = uniform_cost_search(goals, ii, g, cost);

        auto t2 = high_resolution_clock::now();
        duration<double, std::micro> ms_double = t2 - t1;
        cout<<"iteration "<<ms_double.count()<<","<<lS<<","<<lD<<endl;

        MatDistance[i++]=distgoals;
    }
    cout<<"Fin iter sur edge"<<endl;
    return MatDistance;
}
/*
 * The main programme
 */

int main() {
    Graph_t gin;
    Graph_t g;
    readGraphMLFile(gin, "/Users/ksalamatian/processed.graphml");
    k_core(gin,g, 2);
    typedef graph_traits<Graph_t>::edge_iterator edge_iter;
    typedef graph_traits<Graph_t>::vertex_iterator vertex_iter;
    edge_iter ei, ei_end;
    vertex_iter vi, vi_end;
    double delta_t = 0.01;
    tie(ei, ei_end) = edges(g);
    vector<Edge> edges(ei,ei_end);
    tie(vi, vi_end) = vertices(g);
    vector<Vertex> vertices(vi, vi_end);
    IndexMap indexV = get(vertex_index,g);
    vector<vector<int>> graphVertex;    //graphVertex[i]= vecteur de tout les voisins de i
    graphVertex.resize(num_vertices(g));
    map<pair<int, int>, double> cost;   // clé: l'arrete= source, desitination    val: le weight
    std::queue<pair<int, int>> ed;     // queue des arretes


    //On construit graphVertex
    parallel_for(blocked_range<int>(0, num_vertices(g),10000),
                 [&](blocked_range<int> r)
                 {
                     for (int i=r.begin(); i<r.end(); ++i) {
                         pair<AdjacencyIterator, AdjacencyIterator> nA = adjacent_vertices(vertices[i], g);
                         for (; nA.first != nA.second; ++nA.first) {
                             graphVertex[indexV[vertices[i]]].push_back(indexV[*nA.first]);
                             //cout<<"Construction du graphVertex"<<endl;
                         }
                         //cout<<"Size: "<<indexV[vertices[i]]<<" est: "<< graphVertex[indexV[vertices[i]]].size()<<endl;
                         }
                 });

    //On construit cost et ed
    parallel_for(blocked_range<int>(0,edges.size(),10000),
                 [&](blocked_range<int> r) {
                     for (int i = r.begin(); i < r.end(); ++i) {
                         Vertex s = source(edges[i], g);
                         Vertex d = target(edges[i], g);
                         cost[make_pair(indexV[s], indexV[d])] = g[edges[i]].distance;
                         //cout<<"Construction du cost et ed"<<endl;

                     }
                 });


    map<pair<int, int>, double>::iterator itr;
    for(itr = cost.begin(); itr!= cost.end(); ++itr){
        ed.push(itr->first);
        //cout << (itr->first).first <<"   "<< (itr->first).second <<endl;
    }

    cout<<"Fin construction"<<endl;

    parallel_for(blocked_range<int>(0,edges.size(),10000),
                 [&](blocked_range<int> r)
                 {
                     try {
                         Graph_t gcopy;
                         double averageTime =0.0;
                         double totalTime=0;
                         copy_graph(g,gcopy);
                         for (int i=r.begin(); i<r.end(); ++i) {
                             auto t1 = high_resolution_clock::now();

                             pair<int, int> p = ed.front();
                             ed.pop();
                             vector<std::vector<double>> MatDist = calcul_chemin(p.first, p.second, graphVertex, cost);
                             int sS = graphVertex[p.first].size();
                             int sD = graphVertex[p.second].size();
                             double uS = (double) 1 / sS;
                             double uD = (double) 1 / sD;
                             vector<double> distributionA(sS, uS);
                             vector<double> distributionB(sS, uD);
                             double wd = compute_EMD(distributionA, distributionB, MatDist);
                             auto e = add_edge(p.first, p.second, g).first;
                             g[e].curvature=  (1 - wd) ;
                             cout<<"<curvature: "<< g[e].curvature<<endl;
                             auto t2 = high_resolution_clock::now();
                             duration<double, std::micro> ms_double = t2 - t1;
                             totalTime +=ms_double.count();
                             averageTime =totalTime/(i-r.begin()+1);
                             cout<<averageTime<<","<<i-r.begin()<<","<<r.begin()<<endl;
                         }
                     }catch (const tbb::captured_exception &exc) {
                         cout << "Type: " << typeid(exc).name() << "\n";
                     }
                 });
    return 0;
}





