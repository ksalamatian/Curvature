//
// Created by ksalamatian on 02/01/2022.
//

#include "emd.h"
using namespace std;

int main(){
    vector<vector<double>> M{{1,0,1},{2,1,0},{2,2,1}};
    vector<double> a{0.5,0,0.5},b{0,0.5,0.5};
    for(int i=0;i<100;i++){
        double wd = compute_EMD(a, b, M);
        cout<<wd<<endl;
    }
}