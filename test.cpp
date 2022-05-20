//
// Created by ksalamatian on 02/01/2022.
//

#include "emd.h"
using namespace std;

int main(){
    vector<vector<double>> M{{1,0,1,0},{1,1,0,0},{1,1,0,0}, {1,0,1,0}};
    vector<double> a{0.25,0.25,0.25, 0.25},b{0.25,0.25,0.25, 0.25};
    double wd = compute_EMD(a, b, M);
    cout<<wd<<endl;
}