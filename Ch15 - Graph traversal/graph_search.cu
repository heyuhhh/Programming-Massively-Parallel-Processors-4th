#include <cstdio>
#include <iostream>

#define MAX_VERTEX 2000
#define SOURCE_VERTEX 0
#define SECTION_SIEZ 16
#define PRIVATE_CAPACITY 1024

class Graph {
private:
    int vertex_num = 0;
    int edge_num = 0;
    int **joint_matrix;
    int *dest;
    int *row_ptr;
private:
    void show_dense_matrix() const {
        printf(" ===================   Origin Matrix   ===================>\n");
        for (int r = 0; r < this->vertex_num; ++r){
            for(int c = 0; c < this->vertex_num; ++c){
                printf("%.1d ", this -> joint_matrix[r][c]);
            }
            printf("\n");
        }
        printf("\n");
    }
    void show_csr_matrix() const{
        printf(" ===================   CSR   ===================>\n");
        printf("\nCSR Dest ===> ");
        for (int i = 0; i < this->edge_num; ++i){
            printf("%d ", this-> dest[i]);
        }
        printf("\nCSR Row_ptr ===> ");
        for (int i = 0; i < this -> vertex_num + 1; ++i){
            printf("%d ", this -> row_ptr[i]);
        }
        printf("\n\n");
    }
public:
    Graph(const int &vertex_num, const float &sparse_ratio):vertex_num(vertex_num) {
        joint_matrix = new int*[this->vertex_num];
        for (int i = 0; i <vertex_num; i++) {
           joint_matrix[i] = new int[this->vertex_num];
        }
        for (int i = 0; i < vertex_num; i++) {
            for (int j = 0; j < vertex_num; j++) {
                if (rand() / (float) RAND_MAX <= sparse_ratio && i != j) {
                   joint_matrix[i][j] = 1;
                    ++edge_num;
                } else {
                   joint_matrix[i][j] = 0;
                }
            }
        }

        dest = new int[edge_num];
        row_ptr = new int[vertex_num + 1];

        row_ptr[0] = 0;
        int nonzeros_all = 0;
        for (int i = 0; i < vertex_num; i++) {
            for (int j = 0; j < vertex_num; j++) {
                if (joint_matrix[i][j]) {
                    dest[nonzeros_all] = j;
                    ++nonzeros_all;
                }
            }
            row_ptr[i + 1] = nonzeros_all;
        }
    }
    ~Graph() {
        delete []row_ptr;
        delete []dest;
        for (int i = 0; i < vertex_num; ++i){
            delete []joint_matrix[i];
        }        
    }
    void show(int type = 0) const{
        switch (type) {
            case 0: this -> show_dense_matrix();
                break;
            case 1: this -> show_csr_matrix();
                break;
            default:
                break;
        }
    }
    int get_edge_num() const{
        return edge_num;
    }
    int** get_joint_matrix() const{
        return joint_matrix;
    }
    int* get_dest() const{
        return dest;
    }
    int* get_row_ptr() const{
        return row_ptr;
    }
};

__global__
void bfs_kernel(int *pre_frontier_dev, int *cur_frontier, int *pre_size_dev, int *cur_size_dev, int *dist_dev,
                int *dest_dev, int *row_ptr_dev, int *visited_dev) {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    // data privilization
    __shared__ int local_data[PRIVATE_CAPACITY];
    __shared__ int local_num;
    if (threadIdx.x == 0) {
        local_num = 0;
    }
    __syncthreads();
    if (n < pre_size_dev[0]) {
        int u = pre_frontier_dev[n];
        for (int i = row_ptr_dev[u]; i < row_ptr_dev[u + 1]; i++) {
            int v = dest_dev[i];
            if (atomicAdd(&visited_dev[v], 1) == 0) {
                dist_dev[v] = dist_dev[u] + 1;
                /*
                    优化：
                    写入共享内存时线程束可能还会存在竞争，可以将共享内存修改为二级索引的方式
                    不改变空间大小的同时减少冲突，这样就只会在二级索引的时候出现冲突
                */
                if (local_num < PRIVATE_CAPACITY) {
                    int tmp = atomicAdd(&local_num, 1);
                    local_data[tmp] = v;
                } else {
                    int cur = atomicAdd(&cur_size_dev[0], 1);
                    cur_frontier[cur] = v;
                }
            }
        }
    }
    __syncthreads();
    // shared data to global data
    __shared__ int startIdx;
    if (threadIdx.x == 0) {
        startIdx = atomicAdd(&cur_size_dev[0], local_num);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < local_num; i += blockDim.x) {
        cur_frontier[startIdx + i] = local_data[i];
    }
}

/*
    Taken from:
    https://github.com/Syencil/Programming_Massively_Parallel_Processors/blob/master/PMPP/GraphSearch.cu
*/
void insert_into_dist(int source, int *frontier, int *frontier_size){
    frontier[(*frontier_size)++] = source;
}
void BFS_sequential(const int &source, const int *row_ptr, const int *dest, int *dist){
    int frontier[2][MAX_VERTEX];
    int *pre_froniter = &frontier[0][0];
    int *cur_frontier = &frontier[1][0];
    int pre_size = 0;
    int cur_size = 0;
    // 初始化配置
    insert_into_dist(source, pre_froniter, &pre_size);
    dist[source] = 0;
    while (pre_size > 0){
        // 遍历所有存储的节点
        for (int i = 0; i < pre_size; ++i){
            int cur_vertex = pre_froniter[i];
            // 遍历当前节点中的所有分支
           for (int j = row_ptr[cur_vertex]; j < row_ptr[cur_vertex+1]; ++j){
                if (dist[dest[j]] == -1){
                    insert_into_dist(dest[j], cur_frontier, &cur_size);
                    dist[dest[j]] = dist[cur_vertex] + 1;
                }
            }
        }
        // cur赋值给pre，重置cur
        std::swap(pre_froniter, cur_frontier);
        pre_size = cur_size;
        cur_size = 0;
    }
}

void BFS_Bqueue(int S, int *dest, int *row_ptr, int *dist, int edge_num) {
    int *pre_frontier = new int[MAX_VERTEX];
    int *cur_frontier = new int[MAX_VERTEX];
    int *visited = new int[MAX_VERTEX];

    // init
    memset(visited, 0, sizeof(int) * MAX_VERTEX);
    memset(dist, -1, sizeof(int) * MAX_VERTEX);    

    int pre_size = 1, cur_size = 0;
    pre_frontier[0] = S;
    dist[S] = 0;
    visited[S] = 1;

    int *cur_size_dev, *pre_size_dev;
    int *dest_dev, *dist_dev, *row_ptr_dev, *visited_dev;
    int *cur_frontier_dev, *pre_frontier_dev;

    cudaMalloc((void**)&cur_size_dev, sizeof(int));
    cudaMalloc((void**)&pre_size_dev, sizeof(int));
    cudaMalloc((void**)&dest_dev, sizeof(int) * edge_num);
    cudaMalloc((void**)&dist_dev, sizeof(int) * MAX_VERTEX);
    cudaMalloc((void**)&cur_frontier_dev, sizeof(int) * MAX_VERTEX);
    cudaMalloc((void**)&pre_frontier_dev, sizeof(int) * MAX_VERTEX);
    cudaMalloc((void**)&row_ptr_dev, sizeof(int) * edge_num);
    cudaMalloc((void**)&visited_dev, sizeof(int) * MAX_VERTEX);
    cudaMemcpy(cur_size_dev, &cur_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pre_size_dev, &pre_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dest_dev, dest, sizeof(int) * edge_num, cudaMemcpyHostToDevice);
    cudaMemcpy(dist_dev, dist, sizeof(int) * MAX_VERTEX, cudaMemcpyHostToDevice);
    cudaMemcpy(cur_frontier_dev, cur_frontier, sizeof(int) * MAX_VERTEX, cudaMemcpyHostToDevice);
    cudaMemcpy(pre_frontier_dev, pre_frontier, sizeof(int) * MAX_VERTEX, cudaMemcpyHostToDevice);
    cudaMemcpy(visited_dev, visited, sizeof(int) * MAX_VERTEX, cudaMemcpyHostToDevice);  
    cudaMemcpy(row_ptr_dev, row_ptr, sizeof(int) * edge_num, cudaMemcpyHostToDevice); 

    while (pre_size > 0) {
        dim3 blockDim(SECTION_SIEZ);
        dim3 gridDim((MAX_VERTEX + SECTION_SIEZ - 1) / SECTION_SIEZ);
        bfs_kernel<<<gridDim, blockDim>>>(pre_frontier_dev, cur_frontier_dev, pre_size_dev, cur_size_dev, dist_dev,
                                            dest_dev, row_ptr_dev, visited_dev);
        cudaMemcpy(pre_frontier_dev, cur_frontier_dev, sizeof(int) * MAX_VERTEX, cudaMemcpyDeviceToDevice);
        cudaMemcpy(pre_size_dev, cur_size_dev, sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemset(cur_frontier_dev, 0, sizeof(int) * MAX_VERTEX);
        cudaMemset(cur_size_dev, 0, sizeof(int));
        cudaMemcpy(&pre_size, pre_size_dev, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(pre_frontier, pre_frontier_dev, sizeof(int) * MAX_VERTEX, cudaMemcpyDeviceToHost);
        // printf("%d\n", pre_size);
        // for (int i = 0; i < pre_size; i++) {
        //     printf("%d ", pre_frontier[i]);
        // }
        // printf("\n");
    }

    cudaMemcpy(dist, dist_dev, sizeof(int) * MAX_VERTEX, cudaMemcpyDeviceToHost);

    int *dist_output = new int[MAX_VERTEX];
    memset(dist_output, -1, sizeof(int) * MAX_VERTEX);
    BFS_sequential(SOURCE_VERTEX, row_ptr, dest, dist_output);

    for (int i = 0; i < MAX_VERTEX; i++) {
        if (dist[i] != dist_output[i]) {
            printf("i: %d, dist[i] = %d, dist_output[i] = %d\n", i, dist[i], dist_output[i]);
            exit(1);
        }
    }
    printf("比对无差异\n");
}

int main() {
    Graph G(MAX_VERTEX, 0.2);

    // G.show(0);
    // G.show(1);

    int *dest = G.get_dest();
    int *row_ptr = G.get_row_ptr();
    int *dist = new int[MAX_VERTEX];

    BFS_Bqueue(SOURCE_VERTEX, dest, row_ptr, dist, G.get_edge_num());

    delete[] dist;
    return 0;
}