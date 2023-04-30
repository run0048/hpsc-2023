#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
/*
    for(int j=0; j<N; j++) {
      if(i != j) {
	float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
*/
    __m256 xvec = _mm256_load_ps(x);
    __m256 yvec = _mm256_load_ps(y);
    __m256 rxvec = _mm256_setzero_ps();
    __m256 ryvec = _mm256_setzero_ps();
    __m256 xi = _mm256_set1_ps(x[i]);
    __m256 yi = _mm256_set1_ps(y[i]);
    __m256 mask = _mm256_cmp_ps(xvec,xi,_CMP_NEQ_OQ);
    xvec = _mm256_blendv_ps(rxvec,xvec,mask);
    yvec = _mm256_blendv_ps(ryvec,yvec,mask);
    rxvec = _mm256_sub_ps(xi,xvec);
    ryvec = _mm256_sub_ps(yi,yvec);
    __m256 rvec = _mm256_rsqrt_ps(rxvec);
    __m256 one = _mm256_set1_ps(1);
    rvec = _mm256_div_ps(one,rvec);
    float rx[N],ry[N],r[N];
    _mm256_store_ps(rx,rxvec);
    _mm256_store_ps(ry,ryvec);
    _mm256_store_ps(r,rvec);
   for(int j=0; j<N; j++) {
      fx[i] -= rx[j] * m[j] / (r[j] * r[j] * r[j]);
      fy[i] -= ry[j] * m[j] / (r[j] * r[j] * r[j]);
   }  
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
