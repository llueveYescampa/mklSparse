#include <math.h>

void getRowsNnzPerProc(int *rowsPP, int *nnzPP, const int *global_n, const int *global_nnz,  const int *row_Ptr) 
{
    int worldSize=1;

    double nnzIncre = (double) *global_nnz/ (double) worldSize;
    double lookingFor=nnzIncre;
    int startRow=0, endRow;
    int partition=0;

    for (int row=0; row<*global_n; ++row) {    
        if ( (double) row_Ptr[row+1] >=  lookingFor ) { 
            // search for smallest difference
            if (fabs ( lookingFor - row_Ptr[row+1])  <= fabs ( lookingFor - row_Ptr[row])   ) {
                endRow = row;
            } else {
                endRow = row-1;
            } // end if //
            
            rowsPP[partition] = endRow-startRow+1;
            nnzPP[partition]  = row_Ptr[endRow+1] - row_Ptr[startRow];
             
            startRow = endRow+1;
            ++partition;
            if (partition < worldSize-1) {
               lookingFor += nnzIncre;
            } else {
                lookingFor=*global_nnz;
            } // end if //   
        } // end if // 
    } // end for //
} // end of getRowsPerProc //
