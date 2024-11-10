//
// Created by Ryan Gallagher on 11/5/24.
//

#ifndef PROJECT2_TRAVELINGSALESMAN_H
#define PROJECT2_TRAVELINGSALESMAN_H

#include <iostream>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <vector>
#include <cstdio> 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <utility>

FILE *outputFile;

// Structure to hold a path and its corresponding weight
struct PathWeight {
    std::vector<int> path;
    long weight;
};

int loadData(const char *filename, std::vector<float>&latitudes, std::vector<float>&longitudes, std::vector<int>&ids, std::vector<int>&distances);
int readDataSet(FILE *file, std::vector<float>&latitudes, std::vector<float>&longitudes, std::vector<int>&ids, std::vector<int>&distances);
__global__ void populateSymetricDistances(float *, float *, int *, int);
__global__ void greedySearch(int *, int *, long *, bool *, int);
std::vector<PathWeight> getBestPath(const int *, const long *, int, int);
__device__ void calculateWeight(int *, int, int *, long *);
__global__ void recombineSolutions(int *, int, int, int, int *, int *, bool *, long *);
void writeToOutputFile(long *, int, int, int);
void runEvolutionaryAlgorithmOnGPUs(int *, int *, long *, int, int, int, int);

#endif
