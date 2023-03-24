//
// Created by ksalamatian on 17/12/2021.
//

#include <stdio.h>
#include <iostream>
#include <utility>
#include <thread>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphml.hpp>

struct Vertex_info;
struct Edge_info;
typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::undirectedS, Vertex_info, Edge_info > Graph_t;
typedef boost::graph_traits < Graph_t >::vertex_iterator VertexIterator;
typedef boost::graph_traits < Graph_t >::edge_iterator EdgeIterator;
typedef boost::graph_traits < Graph_t >::adjacency_iterator AdjacencyIterator;
typedef boost::graph_traits < Graph_t >::vertex_descriptor Vertex;
typedef boost::graph_traits < Graph_t >::edge_descriptor Edge;
typedef boost::property_map < Graph_t, boost::vertex_index_t >::type IndexMap;

using namespace boost;
using namespace std;

struct Vertex_info {
    string country;
    string name;
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
    double curvature=999.0;
    bool visited=false;
};

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
    dp.property("curvature", get(&Edge_info::curvature, designG));

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

int main() {
    Graph_t g;
    boost::dynamic_properties dp;
    double delta=0.5;
    readGraphMLFile(g, "/data/Curvature/processed1.gml");
    ofstream logFile;
    logFile.open("/data/Curvature/dest.log", ofstream::out);
    auto es = boost::edges(g);
    for (auto eit = es.first; eit != es.second; ++eit) {
        g[*eit].distance += -2*g[*eit].curvature*delta;
        g[*eit].distance=max(g[*eit].distance, 0.0);
        cout<<g[*eit].distance<<","<<g[*eit].curvature<<endl;
        logFile<<source(*eit,g)<<","<<target(*eit,g)<<","<<g[*eit].distance<<","<<g[*eit].curvature<<endl;
    }
    dp.property("asNumber", get(&Vertex_info::asnumber, g));
    //dp.property("pathnum", get(&Vertex_info::pathnum, designG));
    dp.property("Country", get(&Vertex_info::country, g));
    dp.property("Name", get(&Vertex_info::name, g));
    dp.property("asTime", get(&Vertex_info::astime, g));
    dp.property("prefixNum", get(&Vertex_info::prefixnum, g));
    dp.property("prefixAll", get(&Vertex_info::prefixall, g));
    // dp.property("addAll", get(&Vertex_info::addall, designG));
    // dp.property("addNum", get(&Vertex_info::addnum, designG));
    dp.property("count", get(&Edge_info::pathcount, g));
    dp.property("edgeTime", get(&Edge_info::edgetime, g));
    dp.property("weight", get(&Edge_info::weight, g));
    dp.property("distance", get(&Edge_info::distance, g));
    dp.property("curvature", get(&Edge_info::curvature, g));
    ofstream outFile;
    outFile.open("/data/Curvature/processed2.gml", ofstream::out);
    write_graphml(outFile, g, dp, true);
    return 0;
}
