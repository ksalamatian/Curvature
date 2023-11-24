//
// Created by Kave Salamatian on 20/04/2021.
//


#include <iostream>
#include <utility>
#include "curvatureHandler.h"

using namespace boost;
using namespace std;
using namespace code_machina;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::microseconds;





//typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::undirectedS, Vertex_info_BGP, Edge_info_BGP, Graph_info > Graph_t;
//typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::undirectedS, Vertex_info_road, Edge_info_road, Graph_info > Graph_t;
typedef unsigned int uint;

boost::dynamic_properties dp;

/*
 * Read a graphml
 */
void readGraphMLFile (Graph_t& designG, std::string &fileName ) {

    ifstream inFile;
    dp.property("asNumber", get(&Vertex_info_BGP::asnumber, designG));
    dp.property("pathNum", get(&Vertex_info_BGP::pathnum, designG));
    dp.property("Country", get(&Vertex_info_BGP::country, designG));
    dp.property("Name", get(&Vertex_info_BGP::name, designG));
    dp.property("asTime", get(&Vertex_info_BGP::astime, designG));
    dp.property("prefixNum", get(&Vertex_info_BGP::prefixnum, designG));
    dp.property("prefixAll", get(&Vertex_info_BGP::prefixall, designG));
    dp.property("addAll", get(&Vertex_info_BGP::addall, designG));
    dp.property("addNum", get(&Vertex_info_BGP::addnum, designG));
    dp.property("addCount", get(&Edge_info_BGP::addCount, designG));
    dp.property("edgeTime", get(&Edge_info_BGP::edgetime, designG));
    dp.property("weight", get(&Edge_info_BGP::weight, designG));
    dp.property("distance", get(&Edge_info_BGP::distance, designG));
    dp.property("ot", get(&Edge_info_BGP::ot, designG));
    dp.property("curv", get(&Edge_info_BGP::curv, designG));
    dp.property("pathCount", get(&Edge_info_BGP::pathcount, designG));
    dp.property("count", get(&Edge_info_BGP::prefcount, designG));
/*
    dp.property("label", get(&Vertex_info_road::label, designG));
    dp.property("X", get(&Vertex_info_road::X, designG));
    dp.property("Y", get(&Vertex_info_road::Y, designG));
    dp.property("meta", get(&Vertex_info_road::name, designG) );
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
    dp.property("Edge Label", get(&Edge_info_road::edgeLabel, designG));*/
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
        cout << " Type: " << typeid(exc).name() << "\n";
    }
    cout << "Num Vertices: " << num_vertices(designG) << endl;
    cout << "Num Edges: " << num_edges(designG) << endl;
    inFile.close();
}


int main(int argc, char **argv)  {
    Graph_t *g=new Graph_t, *gin=new Graph_t, *ginter;
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
    string pfilename=path+"/"+filename;
    readGraphMLFile(*gin,pfilename );
    int numIteration=2000;
//    string inFilename="/data/Curvature/processed.0.gml";

//    readGraphMLFile(gin, inFilename);
//    VertexIterator v,vend;

    k_core2(*gin,*g, 2);

    double oldRescaling=1.0;
    boost::dynamic_properties dpout;
    long block;
    bool active=true;
    double x=0.0;
    double y=0.0;
    double z=0.0;

    dpout.property("block", get(&VertexType::block, *g));
    dpout.property("x", get(&VertexType::x, *g));
    dpout.property("y", get(&VertexType::y, *g));
    dpout.property("z", get(&VertexType::z, *g));
    dpout.property("weight", get(&EdgeType::weight, *g));
    dpout.property("distance", get(&EdgeType::distance, *g));
    dpout.property("ot", get(&EdgeType::ot, *g));
    dpout.property("curv", get(&EdgeType::curv, *g));

/*        dpout.property("label", get(&Vertex_info_road::label, *g));
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
        dpout.property("curv", get(&Edge_info_road::curv, *g));*/

    map<double, double> attribute_double2double1,attribute_double2double2;
    associative_property_map<map<double, double>> avgCurv_map(attribute_double2double1);
    associative_property_map<map<double, double>> stdCurv_map(attribute_double2double2);
    dpout.property("avgCurv", avgCurv_map);
    dpout.property("stdCurv",stdCurv_map);
    ricci_flow(g, numIteration, iterationIndex,path);
    return 0;
}