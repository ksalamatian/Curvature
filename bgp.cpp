//
// Created by Kave Salamatian on 20/04/2021.
//


#include <iostream>
#include <utility>
#include "curvatureHandler.h"
#include "GraphSpecial.h"

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

    boost::dynamic_properties dp;

    dp = gettingProperties<Graph_t,VertexType,EdgeType>(designG);


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

struct CablePredicate {// both edge and vertex
    typedef typename graph_traits<Graph_t>::vertex_descriptor Vertex;
    typedef typename graph_traits<Graph_t>::edge_descriptor Edge;

    explicit CablePredicate(Graph_t *g, string cableName): g(g),cableName(cableName){};
    CablePredicate()= default;
    bool operator()(Edge e) const      {return (*g)[e].cableName!=cableName;}
    bool operator()(Vertex vd) const { return true; }
    Graph_t *g;
    string cableName;
};

using CableFiltered_Graph_t = boost::filtered_graph<Graph_t, CablePredicate, CablePredicate>;


int main(int argc, char **argv)  {
    Graph_t *g=new Graph_t, *gin=new Graph_t, *ginter, *g2=new Graph_t;
    string filename, path,filter;
    int iterationIndex=0;
    AlgoType algo;
    if( argc > 2 ) {
        string command1(argv[1]);
        if (command1 == "-P") {
            path=string(argv[2]);
        }
        string command2(argv[3]);
        if (command2 =="-F")
            filename= string(argv[4]);
        string command3(argv[5]);
        if (command3 =="-A")
            algo=MSMD;
        if (string(argv[6])=="SSSD")
            algo=SSSD;
        if (string(argv[6])=="SSMD")
            algo=SSMD;
        if (string(argv[6])=="MSMD")
            algo=MSMD;
        string command4(argv[7]);
        if (command4 =="-f")
            filter=string(argv[8]);
    }


    string pfilename=path+"/"+filename;
    readGraphMLFile(*gin,pfilename );
    k_core2(*gin,*g, 2);
    CablePredicate predicate(g,filter);
    CableFiltered_Graph_t fg(*g, predicate, predicate);
    copy_graph(fg,*g2);
    int negin=num_edges(*g), necable=num_edges(*g2);
    cout<<negin<<","<<necable<<endl;


    double oldRescaling=1.0;
    int numIteration=20;
//    ricci_flow(g, numIteration, iterationIndex,path, algo);
    ricci_flow(g2, numIteration,iterationIndex,path,algo);

    return 0;

}