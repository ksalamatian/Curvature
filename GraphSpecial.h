#include <iostream>

#ifndef CURVATURE_GRAPHSPECIAL_H
#define CURVATURE_GRAPHSPECIAL_H
#include <string>


using namespace boost;
using namespace std;

class Vertex_Prot{
public:
    long resseq;
    string resname;
    string chain;
};

class Edge_Prot{
public:
    long lweight;
};

class Vertex_Regular {
public:
    long block;
    bool active=true;
    double x=0.0;
    double y=0.0;
    double z=0.0;
    string name;
};

class Edge_Regular {
};

struct Graph_Regular{
    std::string name;
    double sumDist=0.0;
    double avgDist=1.0;
    double stdDist=0.0;
    double sumCurv=0.0;
    double avgCurv=0.0;
    double stdCurv=0.0;
    double rescaling=0.0;
    double rstdDist=0.0;

};

struct Vertex_info_road {
    double X=0;
    double Y=0;
    float x=0;
    float y=0;
    std::string name;
    double lat=0;
    double longi=0;
    std::string label="";
    std::string country="";
    double size=0;
    long r=0,g=0,b=0;
    long degree=0;
    long cluster=0;
    double eccentricity=0.0,closnesscentrality=0.0, harmonicclosnesscentrality=0.0, betweenesscentrality=0.0;
    bool active=true;
};

struct Edge_info_road {
    std::string id;
    double dist=1.0;
    double odistance=1.0;
    double distance=1.0;
    long weight=1;
    double oot=0.0;
    double ot=0.0;
    double ocurv=0.0;
    double curv=0.0;
    std::string edgeLabel;
    double delta=0.0;
    int r,g,b;
    bool active=true;
    bool surgery=true;
    double ratio;
    double aratio;
    double pdistance;
};



struct Vertex_info_BGP {
    std::string Country;
    std::string name;
    int prefixNum;
    int prefixAll;
    int asTime;
    int addAll;
    int addNum;
    std::string asNumber;
    int pathNum;
    bool active=true;
};

struct Edge_info_BGP {
    int pathCount;
    int edgeTime;
    int count;
    int prefcount;
    int addCount=0;
    string cableName;
    string id;
    double oot=0.0;
    double ot=0.0;
    double ocurv=0.0;
    double curv=0.0;
    double dist=1.0;
    double odistance=1.0;
    double distance=1.0;
    long weight=1;
};

typedef Vertex_info_BGP VertexSpecial;
typedef Edge_info_BGP EdgeSpecial;
//typedef Vertex_Regular VertexSpecial;
//typedef Edge_Regular EdgeSpecial;

//typedef Vertex_info_road VertexSpecial;
//typedef Edge_info_road  EdgeSpecial;
typedef Graph_Regular GraphSpecial;


template <class myGraph, class myVertex, class myEdge>
boost::dynamic_properties gettingProperties(myGraph& g) {

    boost::dynamic_properties dpout;

/*    dpout.property("weight",get(&myEdge::lweight, g));
    dpout.property("resseq",get(&myVertex::resseq, g));
    dpout.property("resname",get(&myVertex::resname, g));
    dpout.property("chain",get(&myVertex::chain, g));
    dpout.property("distance", get(&myEdge::distance, g));
    dpout.property("ot", get(&myEdge::ot, g));
    dpout.property("curv", get(&myEdge::curv, g));
*/


/*
    dpout.property("distance", get(&myEdge::distance, g));
    dpout.property("ot", get(&myEdge::ot, g));
    dpout.property("curv", get(&myEdge::curv, g));
    dpout.property("dist",get(&myEdge::dist, g));
    dpout.property("edist",get(&myEdge::edist, g));
    dpout.property("x",get(&myVertex::x, g));
    dpout.property("y",get(&myVertex::y, g));
    dpout.property("z",get(&myVertex::z, g));
    dpout.property("block",get(&myVertex::block, g));*/



/*    dpout.property("label", get(&myVertex::label, g));
    dpout.property("country", get(&myVertex::country, g));
    dpout.property("X", get(&myVertex::X, g));
    dpout.property("Y", get(&myVertex::Y, g));
    dpout.property("meta", get(&myVertex::name, g));
    dpout.property("lat", get(&myVertex::lat, g));
    dpout.property("long", get(&myVertex::longi, g));
    dpout.property("r", get(&myVertex::r, g));
    dpout.property("g", get(&myVertex::g, g));
    dpout.property("b", get(&myVertex::b, g));
    dpout.property("x", get(&myVertex::x, g));
    dpout.property("y", get(&myVertex::y, g));
    dpout.property("size", get(&myVertex::size, g));
    dpout.property("Degré", get(&myVertex::degree, g));
    dpout.property("Modularity Class", get(&myVertex::cluster, g));
    dpout.property("Eccentricity", get(&myVertex::eccentricity, g));
    dpout.property("Closeness Centrality", get(&myVertex::closnesscentrality, g));
    dpout.property("Harmonic Closeness Centrality", get(&myVertex::harmonicclosnesscentrality, g));
    dpout.property("Betweenness Centrality", get(&myVertex::betweenesscentrality, g));
    dpout.property("dist", get(&myEdge::dist, g));
    dpout.property("weight", get(&myEdge::weight, g));
    dpout.property("distance", get(&myEdge::distance, g));
    dpout.property("ot", get(&myEdge::ot, g));
    dpout.property("curv", get(&myEdge::curv, g));
    dpout.property("id", get(&myEdge::id, g));
    dpout.property("ratio", get(&myEdge::ratio, g));
    dpout.property("aratio", get(&myEdge::aratio, g));
    dpout.property("pdistance", get(&myEdge::pdistance, g));*/



    dpout.property("asNumber", get(&myVertex::asNumber, g));
    dpout.property("pathNum", get(&myVertex::pathNum, g));
    dpout.property("Country", get(&myVertex::Country, g));
    dpout.property("Name", get(&myVertex::name, g));
    dpout.property("asTime", get(&myVertex::asTime, g));
    dpout.property("prefixNum", get(&myVertex::prefixNum, g));
    dpout.property("prefixAll", get(&myVertex::prefixAll, g));
    dpout.property("addAll", get(&myVertex::addAll, g));
    dpout.property("addNum", get(&myVertex::addNum, g));

    dpout.property("pathCount", get(&myEdge::pathCount, g));
    dpout.property("addCount", get(&myEdge::addCount, g));
    dpout.property("edgeTime", get(&myEdge::edgeTime, g));
    dpout.property("weight", get(&myEdge::weight, g));
    dpout.property("prefCount", get(&myEdge::prefcount, g));
    dpout.property("count", get(&myEdge::count, g));
    dpout.property("distance", get(&myEdge::distance, g));
    dpout.property("ot", get(&myEdge::ot, g));
    dpout.property("cable",get(&myEdge::cableName,g));
    dpout.property("id",get(&myEdge::id,g));

    dpout.property("curv", get(&myEdge::curv, g));
    dpout.property("oot",get(&myEdge::oot,g));
    dpout.property("ocurv",get(&myEdge::ocurv,g));
    dpout.property("odistance",get(&myEdge::odistance,g));

    map<double, double> attribute_double2double1,attribute_double2double2;
    map<string,string> nameAttribute;
    associative_property_map<map<double, double>> avgCurv_map(attribute_double2double1);
    associative_property_map<map<double, double>> stdCurv_map(attribute_double2double2);
    associative_property_map<map<string,string>> name(nameAttribute);
    dpout.property("avgCurv", avgCurv_map);
    dpout.property("stdCurv",stdCurv_map);
//    dpout.property("name",name);
    dpout.property("name", get(&myVertex::name, g));

    return dpout;
}


#endif //CURVATURE_GRAPHSPECIAL_H
