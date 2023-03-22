
#include <iostream>

#ifndef CURVATURE_GRAPHSPECIAL_H
#define CURVATURE_GRAPHSPECIAL_H
#include <string>


using namespace boost;
using namespace std;

class Vertex_Regular {
public:
    long block;
    bool active=true;
    double x=0.0;
    double y=0.0;
    double z=0.0;
};

class Edge_Regular {
public:
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
    float X=0;
    float Y=0;
    float x=0;
    float y=0;
    std::string name;
    float lat=0;
    float longi=0;
    std::string label="";
    float size=0;
    int r=0,g=0,b=0;
    int degree=0;
    int cluster=0;
    double eccentricity=0.0,closnesscentrality=0.0, harmonicclosnesscentrality=0.0, betweenesscentrality=0.0;
    bool active=true;
};

struct Edge_info_road {
    float dist=1.0;
    double distance=1.0;
    double weight=1.0;
    double ot=0.0;
    double curv=0.0;
    std::string edgeLabel;
    double delta=0.0;
    int r,g,b;
    bool active=true;
    bool surgery=true;
};



struct Vertex_info_BGP {
    std::string Country;
    std::string Name;
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
    int weight= 1;
    int pathCount;
    int edgeTime;
    long count;
    long prefcount;
    double distance=1.0;
    double ot=99.0;
    bool visited=false;
    double curv=99.0;
    int addCount=0;
    bool active=true;
    bool surgery=false;
};


//typedef Vertex_info_BGP VertexSpecial;
//typedef Edge_info_BGP EdgeSpecial;
typedef Vertex_info_road VertexSpecial;
typedef Edge_info_road  EdgeSpecial;
typedef Graph_Regular GraphSpecial;




template <class myGraph>
boost::dynamic_properties gettingProperties(myGraph& g) {

    bool road = true;
    bool bgp = false;


    boost::dynamic_properties dpout;


    if (road) {
        dpout.property("label", get(&Vertex_info_road::label, g));
        dpout.property("X", get(&Vertex_info_road::X, g));
        dpout.property("Y", get(&Vertex_info_road::Y, g));
        dpout.property("meta", get(&Vertex_info_road::name, g));
        dpout.property("lat", get(&Vertex_info_road::lat, g));
        dpout.property("long", get(&Vertex_info_road::longi, g));
        dpout.property("r", get(&Vertex_info_road::r, g));
        dpout.property("g", get(&Vertex_info_road::g, g));
        dpout.property("b", get(&Vertex_info_road::b, g));
        dpout.property("x", get(&Vertex_info_road::x, g));
        dpout.property("y", get(&Vertex_info_road::y, g));
        dpout.property("size", get(&Vertex_info_road::size, g));
        dpout.property("Degré", get(&Vertex_info_road::degree, g));
        dpout.property("Modularity Class", get(&Vertex_info_road::cluster, g));
        dpout.property("Eccentricity", get(&Vertex_info_road::eccentricity, g));
        dpout.property("Closeness Centrality", get(&Vertex_info_road::closnesscentrality, g));
        dpout.property("Harmonic Closeness Centrality", get(&Vertex_info_road::harmonicclosnesscentrality, g));
        dpout.property("Betweenness Centrality", get(&Vertex_info_road::betweenesscentrality, g));
        dpout.property("dist", get(&Edge_info_road::dist, g));
        dpout.property("weight", get(&Edge_info_road::weight, g));
        dpout.property("distance", get(&Edge_info_road::distance, g));
        dpout.property("ot", get(&Edge_info_road::ot, g));
        dpout.property("curv", get(&Edge_info_road::curv, g));
    }

    if (bgp) {
        dpout.property("asNumber", get(&Vertex_info_BGP::asNumber, g));
        dpout.property("pathNum", get(&Vertex_info_BGP::pathNum, g));
        dpout.property("Country", get(&Vertex_info_BGP::Country, g));
        dpout.property("Name", get(&Vertex_info_BGP::Name, g));
        dpout.property("asTime", get(&Vertex_info_BGP::asTime, g));
        dpout.property("prefixNum", get(&Vertex_info_BGP::prefixNum, g));
        dpout.property("prefixAll", get(&Vertex_info_BGP::prefixAll, g));
        dpout.property("addAll", get(&Vertex_info_BGP::addAll, g));
        dpout.property("addNum", get(&Vertex_info_BGP::addNum, g));

        dpout.property("pathCount", get(&Edge_info_BGP::pathCount, g));
        dpout.property("addCount", get(&Edge_info_BGP::addCount, g));
        dpout.property("edgeTime", get(&Edge_info_BGP::edgeTime, g));
        dpout.property("weight", get(&Edge_info_BGP::weight, g));
        dpout.property("count", get(&Edge_info_BGP::prefcount, g));
        dpout.property("distance", get(&Edge_info_BGP::distance, g));
        dpout.property("ot", get(&Edge_info_BGP::ot, g));
        dpout.property("curv", get(&Edge_info_BGP::curv, g));
    }

    map<double, double> attribute_double2double1,attribute_double2double2;
    associative_property_map<map<double, double>> avgCurv_map(attribute_double2double1);
    associative_property_map<map<double, double>> stdCurv_map(attribute_double2double2);
    dpout.property("avgCurv", avgCurv_map);
    dpout.property("stdCurv",stdCurv_map);

    return dpout;
}


#endif //CURVATURE_GRAPHSPECIAL_H
