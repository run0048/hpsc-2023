#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init(int *bucket, int range) {
  int i = threadIdx.x;
  if(i>=range) return;
  bucket[i] = 0;
}

__global__ void add(int *bucket, int *key,  int n) {
  int i = threadIdx.x;
  if(i>=n) return;
  bucket[key[i]]++;
}

int main() {
  int n = 50;
  int range = 5;
  int *key;
  cudaMallocManaged(&key,n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *bucket;
  cudaMallocManaged(&bucket,range*sizeof(int));
  init<<<1,range>>>(bucket,range);
  cudaDeviceSynchronize();
  
  add<<<1,n>>>(bucket,key,n);
  cudaDeviceSynchronize();
  /*
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }*/

 
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(bucket);
  }
}
