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

struct Vertex_Regular {
    bool active=true;
    long block;
    double x=0.0;
    double y=0.0;
    double z=0.0;
};

struct Edge_Regular {
    double dist=1.0;
    float edist=0.0;
    double distance=1.0;
    double ot=1.0;
    double curv=1.0;
    bool active=true;
    bool surgery=false;
};

struct Graph_info {
    double avgCurv=0.0;
    double stdCurv=0.0;
    string name;
};


struct Graph_info;
typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::undirectedS, Vertex_Regular, Edge_Regular, Graph_info > Graph_t;








ofstream logFile, logFile1;
boost::dynamic_properties dp;

/*
 * Read a graphml
 */
void readGraphMLFile (Graph_t& designG, std::string &fileName ) {

    ifstream inFile;

    dp.property("dist", get(&Edge_Regular::dist, designG));
    dp.property("distance", get(&Edge_Regular::distance, designG));
    dp.property("ot", get(&Edge_Regular::ot, designG));
    dp.property("curv", get(&Edge_Regular::curv, designG));
    dp.property("edist",get(&Edge_Regular::edist, designG));
    dp.property("x",get(&Vertex_Regular::x, designG));
    dp.property("y",get(&Vertex_Regular::y, designG));
    dp.property("z",get(&Vertex_Regular::z, designG));
    dp.property("block",get(&Vertex_Regular::block, designG));
    map<double, double> attribute_double2double1,attribute_double2double2;
    map<string,string> nameAttribute;
    associative_property_map<map<double, double>> avgCurv_map(attribute_double2double1);
    associative_property_map<map<double, double>> stdCurv_map(attribute_double2double2);
    associative_property_map<map<string,string>> name(nameAttribute);
    dp.property("avgCurv", avgCurv_map);
    dp.property("stdCurv",stdCurv_map);
    dp.property("name",name);

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
    int numIteration=50;
//    string inFilename="/data/Curvature/processed.0.gml";

//    readGraphMLFile(gin, inFilename);

    K_core2<Graph_t> k_core2;
    k_core2(*gin,*g, 2);
    double oldRescaling=1.0;
    Triangle_checker<Graph_t> triangle_checker;
    GenerateTasks<Graph_t> generateTasks;
    Process<Graph_t> process;
    DistanceCache distanceCache(num_vertices(*g));
    CalcStats<Graph_t> calcStats;
    UpdateDistances<Graph_t> updateDistances;

    if (! triangle_checker(*g,1)){
        cout<<"Triangular Inequality fail on input"<<endl;
    }

    for(int index=iterationIndex;index<iterationIndex+numIteration;index++){
        vector<int> component(num_vertices(*g));
        int numComponent = connected_components(*g, &component[0]);
//        distanceCache. clear();
        cout<<"Index:"<<index<<" ";
        generateTasks(*g,tasksToDo<Graph_t>);
        DistanceCache distanceCache(num_vertices(*g));
        numProcessedVertex=0;
        numProcessedEdge=0;
        int num_core=std::thread::hardware_concurrency();
//int        num_core=1;
        int offset=0;
        int k=0;
        string logFilename=path+"/processed/logFile."+to_string(index)+".log", logFilename1=path+"/processed/dest."+to_string(index)+".log";
        logFile.open(logFilename.c_str(), ofstream::out);
        logFile1.open(logFilename1.c_str(), ofstream::out);
//    num_core=1;
        vector<thread> threads(num_core);
        for (int i=0;i<num_core;i++){
            threads[i]=std::thread(process,i,g,&distanceCache);
        }
        for (int i=0;i<num_core;i++){
            threads[i].join();
        }
        ofstream outFile;
        string outFilename=path+"/processed/processed."+to_string(index+1)+".graphml";
        outFile.open(outFilename.c_str(), ofstream::out);
//        for(boost::tie(v,vend) = vertices(*g); v != vend; ++v) {
//            cout<<(*g)[*v].name<<endl;
//        }
        calcStats(*g, numComponent, component);
        if (updateDistances(*g,oldRescaling)){
            Predicate<Graph_t> predicate{g};
            ginter = new Graph_t();
            Filtered_Graph_t<Graph_t> fg(*g, predicate, predicate);
            copy_graph(fg,*ginter);
            g->clear();
            delete g;
            g=ginter;
        }
        if (! triangle_checker(*g,0)){
            cout<<"Triangular Inequality fail on iteration "<<index<<endl;
        }
        dynamic_properties dpout;

        dpout.property("dist", get(&Edge_Regular::distance, *g));
        dpout.property("ot", get(&Edge_Regular::ot, *g));
        dpout.property("curv", get(&Edge_Regular::curv, *g));
        dpout.property("edist",get(&Edge_Regular::edist, *g));
        dpout.property("block",  get(&Vertex_Regular::block, *g));
        dpout.property("x",get(&Vertex_Regular::x, *g));
        dpout.property("y",get(&Vertex_Regular::y, *g));
        dpout.property("z",get(&Vertex_Regular::z, *g));



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