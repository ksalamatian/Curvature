//
//  emd.h
//  EMD
//
//  Created by ksalamatian on 14/04/2021.
//

#ifndef emd_h
#define emd_h

#include <iostream>
#include <vector>
#include "network_simplex_simple.h"
#include <chrono>
#include <stdio.h>
#define DEBUG 0

using namespace lemon;
// all types should be signed
typedef int64_t arc_id_type; // {short, int, int64_t} ; Should be able to handle (n1*n2+n1+n2) with n1 and n2 the number of nodes (INT_MAX = 46340^2, I64_MAX = 3037000500^2)
typedef double supply_type; // {float, double, int, int64_t} ; Should be able to handle the sum of supplies and *should be signed* (a demand is a negative supply)
typedef double cost_type;  // {float, double, int, int64_t} ; Should be able to handle (number of arcs * maximum cost) and *should be signed*

struct TsFlow {
    int from, to;
    supply_type amount;
};


double compute_EMD(std::vector<double> &weights1, std::vector<double> weights2, std::vector<std::vector<double>> cost){
    typedef FullBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(FullBipartiteDigraph);

    int n1=weights1.size();
    int n2=weights2.size();

    Digraph di(n1, n2);

    NetworkSimplexSimple<Digraph, supply_type, cost_type, arc_id_type> net(di, true, (int)(n1 + n2), (int64_t)(n1*n2));

    net.supplyMap(&weights1[0], n1, &weights2[0], n2);


    arc_id_type idarc = 0;
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            net.setCost(di.arcFromId(idarc), cost[i][j]);
            idarc++;
        }
    }

    // We run the network simplex

    int ret = net.run();
    double resultdist = net.totalCost();
    return resultdist ;
}

#endif /* emd_h */
