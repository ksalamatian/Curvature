//
// Created by ksalamatian on 19/02/2023.
//

#ifndef CURVATURE_GRAPHSPECIAL_H
#define CURVATURE_GRAPHSPECIAL_H
#include <string>
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

struct Graph_info {
    double sumDist=0.0;
    double avgDist=1.0;
    double stdDist=0.0;
    double sumCurv=0.0;
    double avgCurv=0.0;
    double stdCurv=0.0;
    double rescaling=0.0;
    double rstdDist=0.0;
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
    bool active=true;
};

struct Edge_info_BGP {
    int weight= 1;
    int pathcount=0;
    int edgetime=0;
    //int ctime;
    long prefcount=0;
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
//typedef Graph_Regular GraphSpecial;
typedef Vertex_info_road VertexSpecial;
typedef Edge_info_road  EdgeSpecial;
typedef Graph_Regular GraphSpecial;


#endif //CURVATURE_GRAPHSPECIAL_H
