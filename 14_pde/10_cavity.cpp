#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <openacc.h>

using namespace std;
typedef vector<vector<float> > matrix;



int main() {
  const int nx = 162;
  const int ny = 162;
  int nt = 10;
  int nit = 50;
  float dx = 2. / (nx - 1);
  float dy = 2. / (ny -1);
  float dt = 0.01;
  float rho = 1;
  float nu = 0.02;
  vector<float> x(nx);
  vector<float> y(ny);
  for (int i=0; i<nx; i++)
    x[i] = i * dx;
  for (int i=0; i<nx; i++)
    y[i] = i * dy;
  matrix u(ny,vector<float>(nx,0));
  matrix v(ny,vector<float>(nx,0));
  matrix b(ny,vector<float>(nx,0));
  matrix p(ny,vector<float>(nx,0));
  matrix un(ny,vector<float>(nx,0));
  matrix vn(ny,vector<float>(nx,0));
  matrix pn(ny,vector<float>(nx,0));
  for (int n=0; n<nt; n++) {
    auto tic = chrono::steady_clock::now();
    for (int j=1; j<ny-1; j++){
      for (int i=1; i<nx-1; i++){
        b[j][i] = rho * (1 / dt *
          ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
          ((u[j][i+1] - u[j][i-1]) / (2 * dx))*((u[j][i+1] - u[j][i-1]) / (2 * dx)) 
          - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) *
          (v[j][i+1] - v[j][i-1]) / (2 * dx)) -
          ((v[j+1][i] - v[j-1][i]) / (2 * dy))*((v[j+1][i] - v[j-1][i]) / (2 * dy)));
      }
    }
    auto toc = chrono::steady_clock::now();
    double time = chrono::duration<double>(toc -tic).count();
    printf("step=%d: %lf s\n",n,time);
    for (int it=0; it<nit; it++){
      for (int j=1; j<ny; j++){
        for (int i=1; i<nx; i++){
          pn[j][i] = p[j][i];
        }
      }
#pragma omp parallel for
      for (int j=1; j<ny-1; j++){
        for (int i=1; i<nx-1; i++){
          p[j][i] = (dy*dy * (pn[j][i+1] + pn[j][i-1]) +
                           dx*dx * (pn[j+1][i] + pn[j-1][i]) -
                           b[j][i] * dx*dx * dy*dy) / (2 * (dx*dx + dy*dy));
        }
      }
      for (int j=1; j<ny-1; j++){
        p[j][nx-1] = p[j][nx-2];
        p[j][0] = p[j][1];
      }  
      for (int i=1; i<nx-1; i++){
        p[0][i] = p[1][i];
        p[ny-1][i] = p[ny-2][i];
      }
    }
    tic = chrono::steady_clock::now();
    time = chrono::duration<double>(tic -toc).count();
    printf("step=%d: %lf s\n",n,time);
    for (int j=0; j<ny; j++){
      for (int i=0; i<nx; i++){
        un[j][i] = u[j][i];
        vn[j][i] = v[j][i];
      }
    }
    for (int j=1; j<ny-1; j+=8){
      for (int i=1; i<nx-1; i+=8){
        __m256 unvec = _mm256_load_ps(&un[j][i]);
        __m256 uim1 = _mm256_load_ps(&un[j][i-1]);
        __m256 uip1 = _mm256_load_ps(&un[j][i+1]);
        __m256 ujm1 = _mm256_load_ps(&un[j-1][i]);
        __m256 pip1 = _mm256_load_ps(&p[j][i+1]);
        __m256 pim1 = _mm256_load_ps(&p[j][i-1]);
        __m256 ujp1 = _mm256_load_ps(&un[j+1][i]);
        __m256 uvec;
        __m256 vnvec = _mm256_load_ps(&vn[j][i]);
        __m256 vim1 = _mm256_load_ps(&vn[j][i-1]);
        __m256 vip1 = _mm256_load_ps(&vn[j][i+1]);
        __m256 pjp1 = _mm256_load_ps(&p[j+1][i]);
        __m256 pjm1 = _mm256_load_ps(&p[j-1][i]);
        __m256 vjm1 = _mm256_load_ps(&vn[j-1][i]);
        __m256 vjp1 = _mm256_load_ps(&vn[j+1][i]);
        __m256 vvec;
        uvec = unvec - unvec * dt / dx * (unvec - uim1)
                               - unvec * dt / dy * (unvec - ujm1)
                               - dt / (2 * rho * dx) * (pip1 - pim1)
                               + nu * dt / dx*dx * (uip1 - 2 * unvec + uim1)
                               + nu * dt / dy*dy * (ujp1 - 2 * unvec + ujm1);
        vvec = vnvec - vnvec * dt / dx * (vnvec - vim1)
                               - vnvec * dt / dy * (vnvec - vjm1)
                               - dt / (2 * rho * dx) * (pjp1 - pjm1)
                               + nu * dt / dx*dx * (vip1 - 2 * vnvec + vim1)
                               + nu * dt / dy*dy * (vjp1 - 2 * vnvec + vjm1);
        _mm256_store_ps(&u[j][i],uvec);
        _mm256_store_ps(&v[j][i],vvec);
      }
    }
#pragma acc loop
    for (int j=1; j<ny-1; j++){
      u[j][0] = 0;
      u[j][nx-1] = 0;
      v[j][0] = 0;
      v[j][nx-1] = 0;
    } 
#pragma acc loop
    for (int i=1; i<ny-1; i++){
      u[0][i] = 0;
      u[nx-1][i] = 1;
      v[0][i] = 0;
      v[nx-1][i] = 0;
    }
    toc = chrono::steady_clock::now();
    time = chrono::duration<double>(toc -tic).count();
    printf("step=%d: %lf s\n",n,time);
  }

}
