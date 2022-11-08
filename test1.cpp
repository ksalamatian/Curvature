//
// Created by ksalamatian on 16/06/2022.
//
#include <iostream>
#include "LruCache.h"
class DistanceCache {
private:
    ThreadSafeScalableCache<long, double> distanceCache;
public:
    int cacheSize;
    std::atomic<long> hit{0};
    std::atomic<long> miss{0};
    DistanceCache(size_t maxSize, size_t numShards):distanceCache(maxSize, numShards), cacheSize(maxSize){}
    ~DistanceCache(){
        distanceCache.clear();
//        delete &distanceCache;
    }
    std::pair<bool, short int> insert(const int src, int  dst, double value){
        long key=((long)src<<32)+dst;
        distanceCache.insert(key,value);
        key=((long)dst<<32)+src;
        return distanceCache.insert(key,value);
    }

    double find(const int src, const int dst) {
        // return the distance if he finds it and -1 if not
        ThreadSafeScalableCache<long, double>::ConstAccessor ac;
        long key=((long)src<<32)+dst;
        if (distanceCache.find(ac,key)){
            hit++;
            return *ac;
        }
        key=((long)dst<<32)+src;
        if (distanceCache.find(ac,key)){
            hit++;
            return *ac;
        }
        miss++;
        return -1;
    }

    int size(){
        return distanceCache.size();
    }

    void clear() {
        distanceCache.clear();
    }

};

int main(){
    int k,l;
    DistanceCache distanceCache(1000000,1);

    for (int j=0;j<10;j++){
        auto t1 =std::chrono::high_resolution_clock::now(), t2=t1;
        std::chrono::duration<double, std::micro> ms_double;
        double val=0.0;
        for(int i=0; i<5000000;i++){
            k= rand() % 50000;
            l= rand() % 50000;
            distanceCache.insert(k,l,val++);
        }
        t2=std::chrono::high_resolution_clock::now();
        ms_double=t2-t1;
        std::cout <<ms_double.count()/5000000 <<std::endl;
        distanceCache.clear();
    }
}