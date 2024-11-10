#include "TravelingSalesman.h"

// Load data from file into vectors
int loadData(const char *filename, std::vector<float>&latitudes, std::vector<float>&longitudes, std::vector<int>&ids, std::vector<int>&distances) {
    FILE *cities;
    if ((cities = fopen(filename, "r")) == NULL) {
        std::cerr << "Could not open file " << filename << std::endl;
        return false;
    }
    int numCities = readDataSet(cities, latitudes, longitudes, ids, distances);
    fclose(cities);
    return numCities;
}

int readDataSet(FILE *file, std::vector<float>& latitudes, std::vector<float>& longitudes, std::vector<int>& ids, std::vector<int>& distances) {
    char line[256];
    bool isFullMatrix = false;
    int numCities = -1;

    // First, determine file type and dimension
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "DIMENSION", 9) == 0) {
            sscanf(line, "DIMENSION : %d", &numCities);
        } else if (strncmp(line, "EDGE_WEIGHT_SECTION", 19) == 0) {
            isFullMatrix = true;
            break;
        } else if (strncmp(line, "NODE_COORD_SECTION", 18) == 0) {
            break;
        }
    }

    // full matrix means asymetric problem.
    if (isFullMatrix) {
        // Parsing EDGE_WEIGHT_SECTION as a full distance matrix
        int distance;
        for (int i = 0; i < numCities; ++i) {
            // scan a distance in one at a time, and put it into the distance matrix.
            // if we are going from a city to itself, we just set it to INT_MAX
            for (int j = 0; j < numCities; ++j) {
                fscanf(file, "%d", &distance);
                distances.push_back(i == j ? INT_MAX : distance); // put in int max if it's the city to itself.
            }
        }
    } else {
        // Parsing NODE_COORD_SECTION as lat/long coordinates, and we will call the populate distance matrix function later. 
        int id;
        float latitude, longitude;
        while (fscanf(file, "%d %f %f", &id, &latitude, &longitude) == 3) {
            ids.push_back(id);
            latitudes.push_back(latitude);
            longitudes.push_back(longitude);
        }
    }
    return numCities;
}

// our fancy Kernel function which computes the distances of every city in parallel given some coordinates. 
__global__ void populateSymetricDistances(float *deviceLatitudes, float *deviceLongitudes, int *deviceDistances,int numCities) {
    
    // Calculate the row and column (city indices) this thread will compute
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Row index (city i)
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Column index (city j)

    // Check if the thread is within the bounds of the upper triangle of the distance matrix
    if (i < numCities && j < numCities) {
        if (i != j && i < j) {
            // Calculate the index into the distance matrix
            int index = i * numCities + j;
            int reverse_index = j * numCities + i;

            // Compute Euclidean distance and stuff that into the distance matrix at [i][j] and [j][i]
            float lat_diff = deviceLatitudes[i] - deviceLatitudes[j];
            float long_diff = deviceLongitudes[i] - deviceLongitudes[j];
            int distance = (int)sqrt(lat_diff * lat_diff + long_diff * long_diff);

            // Set both distances to ensure symmetry
            deviceDistances[index] = distance;
            deviceDistances[reverse_index] = distance;
        }
        else if (i == j) {
            int index = i * numCities + j;
            // Set distance to INT_MIN when i == j
            deviceDistances[index] = INT_MAX;
        }
    }
}

__global__ void greedySearch(int *deviceGreedyPaths, int *deviceDistances, long *deviceGreedyWeights, bool *visited, int numCities) {
    // Get the thread ID which corresponds to the starting city
    int startCity = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Ensure the thread is within bounds
    if (startCity >= numCities) return;

    // Initialize visited cities (set all to false) for this thread's starting city
    for (int i = 0; i < numCities; i++) {
        visited[startCity * numCities + i] = false;  // Mark city i as unvisited for startCity
    }

    // Set the starting city as visited and initialize the path
    visited[startCity * numCities + startCity] = true;  // Mark the start city as visited
    deviceGreedyPaths[startCity * numCities] = startCity;  // Set the first city in the path
    deviceGreedyWeights[startCity] = 0;  // Set the weight of the path to 0

    int currentCity = startCity;
    int pathIndex = 1;  // Track position in the path
    int totalWeight = 0; // Track the total weight of the path

    // Perform the greedy search until all cities are visited
    for (int i = 1; i < numCities; i++) {
        int closestCity = -1;
        int minDistance = INT_MAX;

        // Find the closest unvisited city
        for (int j = 0; j < numCities; j++) {
            if (!visited[startCity * numCities + j] && deviceDistances[currentCity * numCities + j] < minDistance) {
                minDistance = deviceDistances[currentCity * numCities + j];
                closestCity = j;
            }
        }

        // Mark the closest city as visited
        visited[startCity * numCities + closestCity] = true;

        // Add the closest city to the path and update the total weight
        deviceGreedyPaths[startCity * numCities + pathIndex] = closestCity;
        totalWeight += minDistance;  // Update total weight
        pathIndex++;
        currentCity = closestCity;
    }

    // add the very last cities weight
    if (numCities > 1)
        totalWeight += deviceDistances[currentCity * numCities + startCity];

    // Store the total path weight for this starting city
    deviceGreedyWeights[startCity] = totalWeight;
}

// Function to get the top X best paths based on weights
// this guy can run on the CPU. 
std::vector<PathWeight> getBestPaths(const int* paths, const long* weights, int numCities, int numBest) {
    // Create a vector of PathWeight structs to store paths and weights together
    std::vector<PathWeight> allPaths;

    // Fill the vector with paths and corresponding weights
    for (int i = 0; i < numCities; i++) {
        std::vector<int> path(numCities);
        for (int j = 0; j < numCities; j++) {
            path[j] = paths[i * numCities + j];
        }
        allPaths.push_back({path, weights[i]});
    }

    // Partial sort to get the top `numBest` elements with the lowest weights
    std::partial_sort(allPaths.begin(), allPaths.begin() + numBest, allPaths.end(), 
        [](const PathWeight &a, const PathWeight &b) {
            return a.weight < b.weight; // Sort by weight in ascending order
        });

    // Get only the top `numBest` paths and weights
    std::vector<PathWeight> bestPaths(allPaths.begin(), allPaths.begin() + numBest);
    return bestPaths;
}

__device__ void calculateWeight(int *solution, int numCities, int *distanceMatrix, long *weight) {
    long w = 0;

    for (int i = 0; i < numCities - 1; i++) {
        int fromCity = solution[i];
        int toCity = solution[(i + 1) % numCities];
        
        // Access the distance between fromCity and toCity
        w += distanceMatrix[fromCity * numCities + toCity];
    }
    
    // Add the distance to complete the tour by returning to the start city
    int lastCity = solution[numCities - 1];
    int firstCity = solution[0];
    w += distanceMatrix[lastCity * numCities + firstCity];
    
    *weight = w;
}

__global__ void recombineSolutions(int *devicePaths, int numCities, int numBestSolutions, int GPUID, int *distanceMatrix, int *kids, bool *found, long *deviceNewWeights){

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= numBestSolutions / 2) // Only process half of the solutions
        return;
    /*
    * The logic of our crossing over is this:
    *   We take our parent A and B iterate through until the crossover point, and just flag all the stuff we've seen already
    *   Then we start from the crossover at A, and put in all the numbers from A which we haven't seen yet in B.
    *   We do this until we've put in crossoverAmount elements from A. Then we do vice versa to create kid B
    *   After this, we recompute the distances and send them back in the kids array
    */

    int *parentA = &devicePaths[idx * numCities];
    int *parentB = &devicePaths[(numBestSolutions - idx - 1) * numCities];
    bool *kidAHas = &found[idx * numCities];
    bool *kidBHas = &found[(numBestSolutions - idx - 1) * numCities];
    int *newKidsA = &kids[idx * numCities];
    int *newKidsB = &kids[(numBestSolutions - idx - 1) * numCities];

    // crossover amount is numcities / (2 * GPUID + 2)
    // this means that for gpu 1, we cross over half, then a fourth and so on. Each GPU makes slightly different results. 
    int crossoverAmount = numCities / (GPUID * 2 + 2);
    int startCrossover = (idx * 31) % numCities; // 31 is a prime number that should help spread out where we are crossing over

    int placedA = 0;
    int placedB = 0;
    int placingIndexA = startCrossover;
    int placingIndexB = startCrossover;
    // Loop through A and B. We just flag all the values we find in A and B before the crossover point.
    // after flagging is done, we look at A, and if that value isn't in B yet, we put it in B and increment placing index. If it was already used, we don't increment and look at next number.
    // once we've done all the numbers in crossover section, we run through A and B one last time, to fill in the rest of the values. 
    for (int i = startCrossover; placedB < crossoverAmount; i++) {
        // Process parentA into newKidsB if not already placed and vice versa
        kidBHas[parentA[i % numCities]] = true;
        newKidsB[placingIndexB++ % numCities] = parentA[i % numCities];
        placedB++;
    }

    for (int i = startCrossover; placedA < crossoverAmount; i++){
        kidAHas[parentB[i % numCities]] = true;
        newKidsA[placingIndexA++ % numCities] = parentB[i % numCities];
        placedA++;  
    }
    // Run through A and B to fill any remaining unplaced items
    for (int i = 0; i < numCities; i++) {
        if (!kidBHas[parentB[i]]){
            kidBHas[parentB[i]] = true;
            newKidsB[placingIndexB++ % numCities] = parentB[i];
        }
        if (!kidAHas[parentA[i]]) {
            kidAHas[parentA[i]] = true;
            newKidsA[placingIndexA++ % numCities] = parentA[i];
        }
    }

    // now our random swapping portion to introduce some mutations
    // we pick a random number between 0 and 1/100 of the city set to be the swap amount
    int swapAmount = idx % (numCities / 100);
    for(int i = 0; i < swapAmount; i++){
        
        int index1 = (swapAmount * idx) % numCities;
        int index2 = (swapAmount * idx * 7) % numCities; // pick 7 just because we need some kind of different number and 7 is a nice ugly prime number
        
        int temp = newKidsA[index1];
        newKidsA[index1] = newKidsA[index2];
        newKidsA[index2] = temp;
        
        temp = newKidsB[index1];
        newKidsB[index1] = newKidsB[index2];
        newKidsB[index2] = temp;
    }

    long newWeightA, newWeightB;
    calculateWeight(newKidsA, numCities, distanceMatrix, &newWeightA);
    calculateWeight(newKidsB, numCities, distanceMatrix, &newWeightB);

    deviceNewWeights[idx] = newWeightA;
    deviceNewWeights[numBestSolutions - idx - 1] = newWeightB;
}

void writeToOutputFile(long *weights, int numBestSolutions, int numCities, int generation) {
    // Write Generation with number of generation
    fprintf(outputFile, "Generation %d: \n", generation);
    
    // Iterate through each solution in weights
    for (int i = 0; i < numBestSolutions; i++) {
        //write in each weight with a comma
        fprintf(outputFile, "%ld, ", weights[i]);
    }

    // Extra newline to separate generations
    fprintf(outputFile, "\n");
}

void runEvolutionaryAlgorithmOnGPUs(int *hostDistances, int *hostBestPaths, long *hostBestWeights, int numCities, int numBestSolutions, int gpuCount, int generations) {

    int threadsPerBlock = 1024;
    dim3 numBlocks((numBestSolutions + threadsPerBlock - 1) / threadsPerBlock, (numBestSolutions + threadsPerBlock - 1) / threadsPerBlock);

    // Define arrays with pointers for each GPU
    int *deviceDistancesArray[gpuCount];
    int *deviceBestPathsArray[gpuCount];
    int *kids[gpuCount];
    bool *found[gpuCount];
    long *deviceNewWeights[gpuCount];
    int *hostEvolvedPaths[gpuCount];
    long *hostEvolvedWeights[gpuCount];
    cudaStream_t *streams = new cudaStream_t[gpuCount];

    // Allocate device memory and initialize CUDA streams only once before the loop
    for (int gpu = 0; gpu < gpuCount; gpu++) {
        cudaSetDevice(gpu);

        // Allocate device memory for distances and initial best paths only once
        cudaMalloc(&deviceDistancesArray[gpu], sizeof(int) * numCities * numCities);
        cudaMemcpy(deviceDistancesArray[gpu], hostDistances, sizeof(int) * numCities * numCities, cudaMemcpyHostToDevice);

        cudaMalloc(&deviceBestPathsArray[gpu], sizeof(int) * numCities * numBestSolutions);
        cudaMemcpy(deviceBestPathsArray[gpu], hostBestPaths, sizeof(int) * numCities * numBestSolutions, cudaMemcpyHostToDevice);

        // Allocate memory for evolving solutions (kids) and weights, and initialize found array
        cudaMalloc(&kids[gpu], sizeof(int) * numCities * numBestSolutions);
        cudaMalloc(&found[gpu], sizeof(bool) * numCities * numBestSolutions);
        cudaMalloc(&deviceNewWeights[gpu], sizeof(long) * numBestSolutions);

        hostEvolvedPaths[gpu] = (int*)malloc(sizeof(int) * numCities * numBestSolutions * 2);
        hostEvolvedWeights[gpu] = (long*)malloc(sizeof(long) * numBestSolutions * 2);
        cudaStreamCreate(&streams[gpu]);
    }

    // Main generations loop
    for (int generation = 0; generation < generations; generation++) {

        // Clear the found array at the beginning of each generation
        for (int gpu = 0; gpu < gpuCount; gpu++) {
            cudaSetDevice(gpu);
            cudaMemset(found[gpu], 0, sizeof(bool) * numCities * numBestSolutions);
        }

        // Launch kernel on each GPU
        for (int gpu = 0; gpu < gpuCount; gpu++) {
            cudaSetDevice(gpu);
            recombineSolutions<<<numBlocks, threadsPerBlock>>>(deviceBestPathsArray[gpu], numCities, numBestSolutions, gpu, deviceDistancesArray[gpu], kids[gpu], found[gpu], deviceNewWeights[gpu]);
        }

        // Copy results back asynchronously
        for (int gpu = 0; gpu < gpuCount; gpu++) {
            cudaSetDevice(gpu);
            cudaDeviceSynchronize();
            cudaMemcpyAsync(hostEvolvedPaths[gpu], kids[gpu], sizeof(int) * numCities * numBestSolutions, cudaMemcpyDeviceToHost, streams[gpu]);
            cudaMemcpyAsync(hostEvolvedWeights[gpu], deviceNewWeights[gpu], sizeof(long) * numBestSolutions, cudaMemcpyDeviceToHost, streams[gpu]);
        }

        // Wait for all memory transfers to complete
        for (int gpu = 0; gpu < gpuCount; gpu++) {
            cudaStreamSynchronize(streams[gpu]);
        }

        // After copying, set the new generation's solutions in deviceBestPathsArray for the next generation
        for (int gpu = 0; gpu < gpuCount; gpu++) {
            cudaSetDevice(gpu);
            cudaMemcpy(deviceBestPathsArray[gpu], kids[gpu], sizeof(int) * numCities * numBestSolutions, cudaMemcpyDeviceToDevice);
            writeToOutputFile(hostEvolvedWeights[gpu], numBestSolutions, numCities, generation);
        }
    }

    // Clean up memory after all generations are complete
    for (int gpu = 0; gpu < gpuCount; gpu++) {
        cudaSetDevice(gpu);
        cudaFree(deviceDistancesArray[gpu]);
        cudaFree(deviceBestPathsArray[gpu]);
        cudaFree(kids[gpu]);
        cudaFree(found[gpu]);
        cudaFree(deviceNewWeights[gpu]);
        free(hostEvolvedPaths[gpu]);
        free(hostEvolvedWeights[gpu]);
        cudaStreamDestroy(streams[gpu]);
    }
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    // Global variables (if preferred to reduce passing multiple arguments)
    std::vector<int> ids;
    std::vector<float> latitudes;
    std::vector<float> longitudes;
    std::vector<int> distances;

    int numCities = loadData(argv[1], latitudes, longitudes, ids, distances);

    if (numCities == -1) {
        std::cerr << "Error reading data from file." << std::endl;
        return 1;
    }

    outputFile = fopen("output.csv", "w");
    fprintf(outputFile, "%s\n", argv[1]);
    


    // store our number of ciites so we can read our code easier
    printf("\nAmount of cities read: %d\n", numCities);

    // get number of geneartions from user.
    int numGenerations = -1;
    while(numGenerations <= 0){
        printf("Enter number of generations: ");
        scanf("%d", &numGenerations);

        if (numGenerations < 0) {
            printf("\nInvalid number of generations. Please enter a positive integer.\n");
        }

        if (numGenerations == 0) {
            printf("\nInvalid number of generations. Please enter a positive integer.\n");
        }
    }

    // Allocate memory on the host (CPU) for the distances
    int *hostDistances = (int*)malloc(sizeof(int) * numCities * numCities);
    int *deviceDistances;
    // allocate memory on the device. It is easier to use a 1D array than a 2D array.
    cudaMalloc(&deviceDistances, sizeof(int) * numCities * numCities);

    // populating our distances
    if(latitudes.size() != 0){ // that means we read in a symetric file, so we must now populate the matrix with the lat and longs.

        float *deviceLatitudes, *deviceLongitudes;
    
        // Allocate memory on the device (GPU) and copy it over
        cudaMalloc(&deviceLatitudes, numCities * sizeof(float));
        cudaMalloc(&deviceLongitudes, numCities * sizeof(float));
        cudaMemcpy(deviceLatitudes, latitudes.data(), numCities * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceLongitudes, longitudes.data(), numCities * sizeof(float), cudaMemcpyHostToDevice);
    
        // if num cities is too small, we'll be in trouble, so we need to cap it in this way.
        int numThreadsPerBlock = (numCities < 32) ? numCities : 32;
        dim3 threadsPerBlock(numThreadsPerBlock, numThreadsPerBlock);
        dim3 numBlocks((numCities + numThreadsPerBlock - 1) / numThreadsPerBlock,(numCities + numThreadsPerBlock - 1) / numThreadsPerBlock);
    
        // Launch kernel
        // our total number of threads becomes numCities * numCities 
        populateSymetricDistances<<<numBlocks, threadsPerBlock>>>(deviceLatitudes, deviceLongitudes, deviceDistances, numCities);

        // Check for errors in kernel execution
        cudaDeviceSynchronize();

        // free our coordinates, since we don't need those anymore
        cudaFree(deviceLatitudes);
        cudaFree(deviceLongitudes);

        // Copy the data from the device to the host
        cudaMemcpy(hostDistances, deviceDistances, sizeof(int) * numCities * numCities, cudaMemcpyDeviceToHost);
    }
    else {
        // Copy the data from the host to the device
        cudaMemcpy(deviceDistances, distances.data(), sizeof(int) * numCities * numCities, cudaMemcpyHostToDevice); // copy our data back over to both distances arrays
        memcpy(hostDistances, distances.data(), sizeof(int) * numCities * numCities);
    }

    printf("\nDistances Calculated!\n");

    // now that we have our distance matrix, we can start the fun stuff. We will start by simply doing a greedy algorithm starting from every single city.
    int *hostGreedyPaths = (int*)malloc(sizeof(int) * numCities * numCities); // the path again needs to be N^2 length, since we are doing N cities per N tours
    // weights is just the resulting weight of a given tour after doing it the greedy way
    long *hostGreedyWeights = (long*)malloc(sizeof(long) * numCities);

    int *deviceGreedyPaths;
    long *deviceGreedyWeights;
    bool *visited;

    cudaMalloc(&deviceGreedyPaths, sizeof(int) * numCities * numCities);
    cudaMalloc(&deviceGreedyWeights, sizeof(long) * numCities);
    cudaMalloc(&visited, sizeof(bool) * numCities * numCities);
    cudaMemset(visited, 0, sizeof(bool) * numCities * numCities);
    cudaMemset(deviceGreedyWeights, 0, sizeof(long) * numCities);
    cudaMemset(deviceGreedyPaths, 0, sizeof(int) * numCities * numCities);

    printf("\nLaunching Greedy Algorithm to find Initial Solutions...\n");
    int numThreadsPerBlockGreedy = 1024;
    dim3 threadsPerBlockGreedy(numThreadsPerBlockGreedy);
    dim3 numBlocksGreedy((numCities + numThreadsPerBlockGreedy - 1) / numThreadsPerBlockGreedy);
    // Launch the kernel
    greedySearch<<<numBlocksGreedy, threadsPerBlockGreedy>>>(deviceGreedyPaths, deviceDistances, deviceGreedyWeights, visited, numCities);
    cudaDeviceSynchronize();

    cudaMemcpy(hostGreedyWeights, deviceGreedyWeights, sizeof(long) * numCities, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostGreedyPaths, deviceGreedyPaths, sizeof(int) * numCities * numCities, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceGreedyPaths);
    cudaFree(deviceGreedyWeights);
    cudaFree(visited); 

    printf("\nLaunching Evolutionary Algorithm...\n");

    // we will use the best tenth of our data set as the candidates to get bred
    // if the set is too small, we'll just use half.
    // make sure numBestSolutions is even by rounding up if needed
    int numBestSolutions = numCities > 30 ? (numCities / 10) : (numCities / 2);
    if (numBestSolutions % 2 != 0) {
        numBestSolutions++; // Make it even if it's odd
    }

    // allocate memory on the host for the best solutions
    int *hostBestPaths = (int*)malloc(sizeof(int) * numCities * numBestSolutions);
    long *hostBestWeights = (long*)malloc(sizeof(long) * numBestSolutions);
    
    // get our vector back after sorting our paths by weight, on CPU.
    // turn this vector back into two arrays, one for paths, and one for weights.
    std::vector<PathWeight> bestPaths = getBestPaths(hostGreedyPaths, hostGreedyWeights, numCities, numBestSolutions);
    for(int i = 0; i < numBestSolutions; i++) {
        memcpy(&hostBestPaths[i * numCities], bestPaths[i].path.data(), sizeof(int) * numCities);
        hostBestWeights[i] = bestPaths[i].weight;
    }

    free(hostGreedyPaths);
    free(hostGreedyWeights);

    printf("\nBeginning Recombination...\n");
    // we have FOUR GPUs to run this on, so let's make each GPU do a slightly different version of the algorithm.
    // We will swap 1/2, 1/4, 1/8, 1/16 of each parent solution with each other according to GPU we run on.
    // then we re combine all the solutions into a new population and keep going. 
    int gpuCount;
    cudaGetDeviceCount(&gpuCount);

    runEvolutionaryAlgorithmOnGPUs(hostDistances, hostBestPaths, hostBestWeights, numCities, numBestSolutions, gpuCount, numGenerations);

    printf("\nEvolutionary Algorithm Complete!\n");

    free(hostBestPaths);
    free(hostBestWeights);
    // Now we can free our memory
    cudaFree(deviceDistances);    
    // Free host memory
    free(hostDistances);

    fclose(outputFile);

    return 0;

}   


