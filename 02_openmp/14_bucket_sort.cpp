#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 

 #pragma omp paralell for
  for (int i=0; i<n; i++)
    bucket[key[i]]++;

  std::vector<int> offset(range,0);
  std::vector<int> onset(range,0);

#pragma omp paralell
{  
#pragma omp for
    for (int i=0; i<range; i++)
      onset[i] = offset[i] + bucket[i];
#pragma omp for
    for (int k=0; k<range; k++)
      for (int i=k; i<range; i++)
	offset[i+1] += onset[k];
}


#pragma omp paralell for
{ 
 for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
}

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
