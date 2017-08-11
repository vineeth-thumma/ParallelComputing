#include "mpi.h"
#include <time.h>
#include <stdio.h>
#include <assert.h>

#define NS_PER_SEC 1E9
#define msgsize 1024*1024
#define niter 10
#define Bytes_PER_MB 1E6


int main(int argc, char *argv[]) {

    int rank, num_procs, i, j;
    double* bufferA;
    double* bufferB;
    struct timespec start_time, end_time;
    double total_time, time1, time2, bw1, bw2;
    char proc_name[MPI_MAX_PROCESSOR_NAME]; int len;
    MPI_Status statusA, statusB;
    MPI_Request requestA, requestB;
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    //printf("Rank: %d, No. of Processes: %d", rank, num_procs);
    time1 = 0; time2 = 0;
    bufferA = (double*)malloc(sizeof(double)*msgsize);
    bufferB = (double*)malloc(sizeof(double)*msgsize);
    
   
   for(i=1; i<num_procs; i++) { 
       if(rank == 0) {
        MPI_Barrier(MPI_COMM_WORLD);  
       
        assert(clock_gettime(CLOCK_MONOTONIC, &start_time) != -1);
        for(j=0; j<niter; j++) {
            
            MPI_Isend(bufferA, msgsize, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requestA);
            MPI_Irecv(bufferB, msgsize, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requestB);
            MPI_Wait(&requestB, &statusB);
            MPI_Isend(bufferB, msgsize, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requestB);
            MPI_Irecv(bufferA, msgsize, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &requestA);
            MPI_Wait(&requestA, &statusA);
        } 
       
        assert(clock_gettime(CLOCK_MONOTONIC, &end_time) != -1);
        total_time = (end_time.tv_sec - start_time.tv_sec + (end_time.tv_nsec - start_time.tv_nsec) / NS_PER_SEC);
        if(i<12) {
            time1 += total_time;
        } 
        else {
            time2 += total_time;
        }
        
        MPI_Get_processor_name(proc_name, &len);
        printf("\n Master- Proc. name is %s for iteration %d",proc_name,i);
        printf("\n Total time taken for iteration %d is %lf seconds \n", i, total_time);
     
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == i) {
            
        for(j=0; j<niter; j++) {
            
            MPI_Irecv(bufferA, msgsize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requestA);
            MPI_Wait(&requestA, &statusA);
            MPI_Isend(bufferA, msgsize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requestA);
            MPI_Irecv(bufferB, msgsize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requestB);
            MPI_Wait(&requestB, &statusB);
            MPI_Isend(bufferB, msgsize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requestB);
        }
        MPI_Get_processor_name(proc_name, &len);
    
        printf("\n Slave- Proc. name is %s for iteration %d",proc_name,i);
    }
   
   }
   
 }
 if(rank ==0) {
        printf("\ntotal time 1-11: %lf",time1);
        printf("\ntotal time 12-23: %lf",time2);
        time1 = time1/(4*niter*11);
        time2 = time2/(4*niter*12);
        bw1 = (8*msgsize)/(time1*Bytes_PER_MB);
        bw2 = (8*msgsize)/(time2*Bytes_PER_MB);
        printf("\n Avg. time taken for iterations 1-11 is %lf seconds and Performance is %lf MB \n", time1, bw1);
        printf("\n Avg. time taken for iterations 12-23 is %lf seconds and Performance is %lf MB\n", time2, bw2);
        }
    MPI_Finalize();

}