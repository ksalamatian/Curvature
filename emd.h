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

typedef unsigned int node_id_type;

enum ProblemType {
    INFEASIBLE,
    OPTIMAL,
    UNBOUNDED,
    MAX_ITER_REACHED
};

using namespace lemon;
// all types should be signed
typedef int64_t arc_id_type; // {short, int, int64_t} ; Should be able to handle (n1*n2+n1+n2) with n1 and n2 the number of nodes (INT_MAX = 46340^2, I64_MAX = 3037000500^2)
typedef double supply_type; // {float, double, int, int64_t} ; Should be able to handle the sum of supplies and *should be signed* (a demand is a negative supply)
typedef double cost_type;  // {float, double, int, int64_t} ; Should be able to handle (number of arcs * maximum cost) and *should be signed*

struct TsFlow {
    int from, to;
    supply_type amount;
};

using namespace std;


int EMD_wrap(vector<double> &X, vector<double> &Y, vector<std::vector<double>> &D, vector<vector<double>> &G,
             vector<double> &alpha, vector<double> &beta, double &cost, int maxIter){
    using namespace lemon;
    uint64_t n, m, cur;
    int n1=X.size();
    int n2=Y.size();

    typedef FullBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(Digraph);
    // Get the number of non zero coordinates for r and c
    n=0;
    for (int i=0; i<n1; i++) {
        double val=X[i];
        if (val>0) {
            n++;
        }else if(val<0){
            return INFEASIBLE;
        }
    }
    m=0;
    for (int i=0; i<n2; i++) {
        double val=Y[i];
        if (val>0) {
            m++;
        }else if(val<0){
            return INFEASIBLE;
        }
    }

    // Define the graph

    vector<uint64_t> indI(n), indJ(m);
    vector<double> weights1(n), weights2(m);
    Digraph di(n, m);
    NetworkSimplexSimple<Digraph,double,double, node_id_type> net(di, true, (int) (n + m), n * m, maxIter);

    // Set supply and demand, don't account for 0 values (faster)

    cur=0;
    for (uint64_t i=0; i<n1; i++) {
        double val=X[i];
        if (val>0) {
            weights1[ cur ] = val;
            indI[cur++]=i;
        }
    }

    // Demand is actually negative supply...

    cur=0;
    for (uint64_t i=0; i<n2; i++) {
        double val=Y[i];
        if (val>0) {
            weights2[ cur ] = -val;
            indJ[cur++]=i;
        }
    }

    net.supplyMap(&weights1[0], (int) n, &weights2[0], (int) m);

    // Set the cost of each edge
    int64_t idarc = 0;
    for (uint64_t i=0; i<n; i++) {
        for (uint64_t j=0; j<m; j++) {
            double val=D[indI[i]][indJ[j]];
            net.setCost(di.arcFromId(idarc), val);
            ++idarc;
        }
    }

    // Solve the problem with the network simplex algorithm

    int ret=net.run();
    uint64_t i, j;
    if (ret==OPTIMAL || ret==MAX_ITER_REACHED) {
        cost = 0;
        Arc a; di.first(a);
        for (; a != INVALID; di.next(a)) {
            i = di.source(a);
            j = di.target(a);
            double flow = net.flow(a);
            cost += flow * D[indI[i]][indJ[j-n]];
            G[indI[i]][indJ[j-n]] = flow;
            alpha[indI[i]] = -net.potential(i);
            beta[indJ[j-n]] = net.potential(j);
        }
    }
    return ret;
}

double compute_EMD(std::vector<double> &weights1, std::vector<double> &weights2, std::vector<std::vector<double>> &cost){
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
            net.setCost(Digraph::arcFromId(idarc), cost[i][j]);
            idarc++;
        }
    }

    // We run the network simplex

    int ret = net.run();
    double resultdist = net.totalCost();
    return resultdist ;
}

#endif /* emd_h */
