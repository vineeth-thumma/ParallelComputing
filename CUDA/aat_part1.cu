// Template for Programming Assignment 2
// Use "module load cuda" to enable compilation with the Nvidia C compiler nvcc
// Use "nvcc -O3" to compile code; this can be done even on OSC login node (does not have a GPU)
// To execute compiled code, you must either use a batch submission to run on a node with GPU
// or obtain an interactive GPU-node by using: qsub -I -l walltime=0:59:00 -l nodes=1:gpus=1

#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#define threshold 1e-8
#define n (4096)
// Change n to 4096 for final testing; 
//#define n (1024)
// n is set to 256 since execution time of single thread template version is excessive
#define TILE_WIDTH 8

void init(void);
void ref(void);
void compare(int N, double *wref, double *w);
__global__ void test_kernel(int N, double *A, double *B, double *C);
double rtclock(void);

double a[n][n],b[n][n],c[n][n],cref[n][n];

int main(){

double clkbegin, clkend, t;
double *Ad,*Bd,*Cd;
int size;

  printf("Matrix Size = %d\n",n);

  init();
  clkbegin = rtclock();
  ref();
  clkend = rtclock();
  t = clkend-clkbegin;
  printf("Seq: Approx GFLOPS: %.1f ; Time = %.3f sec; cref[n/2][n/2-1] = %f; \n",
2.0*n*n*n/t/1e9,t,cref[n/2][n/2-1]);

  
  size = sizeof(double)*n*n;
  cudaMalloc((void **) &Ad,size);
  cudaMalloc((void **) &Bd,size);
  cudaMalloc((void **) &Cd,size);
  cudaMemcpy(Ad,a,size,cudaMemcpyHostToDevice);
  cudaMemcpy(Bd,b,size,cudaMemcpyHostToDevice);
  clkbegin = rtclock();
  
  dim3 dimGrid(n/TILE_WIDTH, n/TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  
  test_kernel<<<dimGrid,dimBlock>>>(n,Ad,Bd,Cd);
  if (cudaDeviceSynchronize() != cudaSuccess) 
    printf ("Error return for test_kernel: Was execution done on a node with a GPU?\n");
  else
  {
   clkend = rtclock();
   t = clkend-clkbegin;
   cudaMemcpy(c,Cd,size,cudaMemcpyDeviceToHost);
   cudaFree(Ad); cudaFree(Bd); cudaFree(Cd);
    printf("GPU: Approx GFLOPS: %.1f ; Time = %.3f sec; c[n/2][n/2-1] = %f; \n",
     2.0*n*n*n/t/1e9,t,c[n/2][n/2-1]);
    printf("Correctness Check for GPU solution:\n");
    compare(n, (double *) c,(double *) cref);
  }
}

__global__ void test_kernel(int N, double *A, double *B, double *C)
{

// Block Index along x & y
int bx = blockIdx.x; int by = blockIdx.y;
//Thread Index along x & y
int tx = threadIdx.x; int ty = threadIdx.y;

// Row & Column in resultant matrix C computed by the thread in the block
int Row = by * TILE_WIDTH + ty;
int Column = bx * TILE_WIDTH + tx;

double Pvalue = 0;

// Accumulate dot product
for (int k=0; k<N; ++k)
    Pvalue += A[Row*N+k]*B[Column*N+k];
    
// write final value to global memory
C[Row*N+Column] = Pvalue;

}

void ref(void)
{
int i,j,k;

  for (i=0;i<n;i++)
   for (j=0;j<n;j++)
    for(k=0;k<n;k++)
      cref[i][j] += a[i][k]*b[j][k];
}

void init(void)
{
int i,j;
for(i=0;i<n;i++)
 for(j=0;j<n;j++) 
 { c[i][j] = 0.0; 
   cref[i][j] = 0.0; 
   a[i][j] = drand48();
   b[i][j] = drand48();
 }
}

void compare(int N, double *wref, double *w)
{
double maxdiff,this_diff;
int numdiffs;
int i,j;
  numdiffs = 0;
  maxdiff = 0;
  for (i=0;i<N;i++)
   for (j=0;j<N;j++)
    {
     this_diff = wref[i*N+j]-w[i*N+j];
     if (this_diff < 0) this_diff = -1.0*this_diff;
     if (this_diff>threshold)
      { numdiffs++;
        if (this_diff > maxdiff) maxdiff=this_diff;
      }
    }
   if (numdiffs > 0)
      printf("%d Diffs found over threshold %f; Max Diff = %f\n",
               numdiffs,threshold,maxdiff);
   else
      printf("No differences found between reference and test versions\n");
}

double rtclock(void)
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}
