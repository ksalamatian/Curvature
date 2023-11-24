//
// Created by Kave Salamatian on 20/04/2021.
//
#include <iostream>
#include <utility>
#include <thread>
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

int main(int argc, char **argv)  {
    Graph_t *g=new Graph_t, *gin=new Graph_t, *ginter;
    string filename, path;
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
        string str=string(argv[6]);
        if (str=="SSSD")
            algo=SSSD;
        if (str=="SSMD")
            algo=SSMD;
        if (str=="MSMD")
            algo=MSMD;
    }

    string pfilename=path+"/"+filename;
    readGraphMLFile(*gin,pfilename );
    int numIteration=100;
    k_core2(*gin,*g, 2);


    double oldRescaling=1.0;
    ricci_flow(g, numIteration, iterationIndex,path, algo);
    return 0;
}




