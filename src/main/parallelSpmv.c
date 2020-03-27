#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "mkl_spblas.h"

#include "real.h"

#include "parallelSpmv.h"

#define REP 1000

int main(int argc, char *argv[]) 
{

    const int root=0;
    int worldRank=0;
    
    #include "parallelSpmvData.h"

    // verifing number of input parameters //
   char exists='t';
   char checkSol='f';
    if (worldRank == root) {
        if (argc < 3 ) {
            printf("Use: %s  Matrix_filename InputVector_filename  [SolutionVector_filename]  \n", argv[0]);     
            exists='f';
        } // endif //
        
        FILE *fh=NULL;
        // testing if matrix file exists
        if((fh = fopen(argv[1], "rb")  )   == NULL) {
            printf("No matrix file found.\n");
            exists='f';
        } // end if //
        
        // testing if input file exists
        if((fh = fopen(argv[2], "rb")  )   == NULL) {
            printf("No input vector file found.\n");
            exists='f';
        } // end if //

        // testing if output file exists
        if (argc  >3 ) {
            if((fh = fopen(argv[3], "rb")  )   == NULL) {
                printf("No output vector file found.\n");
                exists='f';
            } else {
                checkSol='t';
            } // end if //
        } // end if //
        if (fh) fclose(fh);
    } // end if //
    //MPI_Bcast(&exists,  1,MPI_CHAR,root,MPI_COMM_WORLD);
    if (exists == 'f') {
        if (worldRank == root) printf("Quitting.....\n");
        //MPI_Finalize();
        exit(0);
    } // end if //
    //MPI_Bcast(&checkSol,1,MPI_CHAR,root,MPI_COMM_WORLD);

    printf("%s Precision.\n", (sizeof(real) == sizeof(double)) ? "Double": "Single");

    
    reader(&n_global,&nnz_global, &n, 
           &off_proc_nnz,
           &row_ptr,&col_idx,&val,
           &row_ptr_off,&col_idx_off,&val_off,
           argv[1], root);
    
    // ready to start //    

    
    real *w=NULL;
    real *v=NULL; // <-- input vector to be shared later
    //real *v_off=NULL; // <-- input vector to be shared later
    
    
    v     = (real *) malloc(n*sizeof(real));
    w     = (real *) malloc(n*sizeof(real)); 
    //v_off = (real *) malloc((nColsOff)*sizeof(real));

    // reading input vector
    vectorReader(v, &n, argv[2]);

    real alpha = 1.0, beta = 0.0;

    // Descriptor of main sparse matrix properties
    struct matrix_descr descrA;
    // Create matrix descriptor
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

    // Structure with sparse matrix stored in CSR format
    sparse_matrix_t       csrA;



    // Create handle with matrix stored in CSR format
    mkl_sparse_d_create_csr ( &csrA, SPARSE_INDEX_BASE_ZERO,
                                    n_global,  // number of rows
                                    n_global,  // number of cols
                                    (MKL_INT *) row_ptr,
                                    (MKL_INT *) row_ptr+1,
                                    (MKL_INT *) col_idx,
                                    val );


    // Analyze sparse matrix; choose proper kernels and workload balancing strategy
    mkl_sparse_optimize ( csrA );


    // Timing should begin here//
    struct timeval tp;                                   // timer
    double elapsed_time;

    gettimeofday(&tp,NULL);  // Unix timer
    elapsed_time = -(tp.tv_sec*1.0e6 + tp.tv_usec);

    for (int t=0; t<REP; ++t) {
        // Compute y = alpha * A * x + beta * y
        mkl_sparse_d_mv ( SPARSE_OPERATION_NON_TRANSPOSE,(double) alpha,csrA,descrA,(double *)v,(double) beta,(double *)w );
    } // end for //

    gettimeofday(&tp,NULL);
    elapsed_time += (tp.tv_sec*1.0e6 + tp.tv_usec);
    
    int worldSize;
    #pragma omp parallel 
    {
        worldSize=omp_get_num_threads();
    } // end of parallel region //
    
    printf ("---> Time taken by  %d threads %f seconds, GFLOPS: %f\n", worldSize ,  elapsed_time*1.0e-6, (2.0*nnz_global+3.0*n_global)*REP*1.0e-3/elapsed_time);

    // Release matrix handle and deallocate matrix
    mkl_sparse_destroy ( csrA );
   
   
    if (checkSol=='t') {
        real *sol=NULL;
        sol     = (real *) malloc((n)*sizeof(real)); 
        // reading input vector
        vectorReader(sol, &n, argv[3]);
        
        int row=0;
        real tolerance = 1.0e-08;
        if (sizeof(real) != sizeof(double) ) {
            tolerance = 1.0e-02;
        } // end if //
        
        real error;
        do {
            error =  fabs(sol[row] - w[row]) /fabs(sol[row]);
            if ( error > tolerance ) break;
            ++row;
        } while (row < n); // end do-while //
        
        if (row == n) {
            printf("Solution match in rank %d\n",worldRank);
        } else {    
            printf("For Matrix %s, solution does not match at element %d in rank %d   %20.13e   -->  %20.13e  error -> %20.13e, tolerance: %20.13e \n", 
            argv[1], (row+1),worldRank, sol[row], w[row], error , tolerance  );
        } // end if //
        free(sol);    
    } // end if //

    
    #include "parallelSpmvCleanData.h" 
    return 0;    
} // end main() //
