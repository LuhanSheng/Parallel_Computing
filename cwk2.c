//
// Starting code for the MPI coursework.
//
// See lectures and/or the worksheet corresponding to this part of the module for instructions
// on how to build and launch MPI programs. A simple makefile has also been included (usage optional).
//


//
// Includes.
//

// Standard includes.
#include <stdio.h>
#include <stdlib.h>

// The MPI library.
#include <mpi.h>

// Some extra routines for this coursework. DO NOT MODIFY OR REPLACE THESE ROUTINES,
// as this file will be replaced with a different version for assessment.
#include "cwk2_extra.h"


//
// Main.
//
int main( int argc, char **argv )
{
    int i;

    //
    // Initialisation.
    //

    // Initialise MPI and get the rank of this process, and the total number of processes.
    int rank, numProcs;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &numProcs );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank     );

    // Check that the number of processes is a power of 2, but <=256, so the data set, which is a multiple of 256 in length,
    // is also a multiple of the number of processes. If using OpenMPI, you may need to add the argument '--oversubscribe'
    // when launnching the executable, to allow more processes than you have cores.
    if( (numProcs&(numProcs-1))!=0 || numProcs>256 )
    {
        // Only display the error message from one processes, but finalise and quit all of them.
        if( rank==0 ) printf( "ERROR: Launch with a number of processes that is a power of 2 (i.e. 2, 4, 8, ...) and <=256.\n" );

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Load the full data set onto rank 0.
    float *globalData = NULL;
    int globalSize = 0;
    if( rank==0 )
    {
        globalData = readDataFromFile( &globalSize );           // globalData must be free'd on rank 0 before quitting.
        if( globalData==NULL )
        {
            MPI_Finalize();                                     // Should really communicate to all other processes that they need to quit as well ...
            return EXIT_FAILURE;
        }
        //printf("data1:%f\n",globalData[0]);
        printf( "Rank 0: Read in data set with %d floats.\n", globalSize );
    }

    // Calculate the number of floats per process. Note that only rank 0 has the correct value of localSize
    // at this point in the code. This will somehow need to be communicated to all other processes. Note also
    // that we can assume that globalSize is a multiple of numProcs.
    int localSize = globalSize / numProcs;          // = 0 at this point of the code for all processes except rank 0.

    // Start the timing now, after the data has been loaded (will only output on rank 0).
    double startTime = MPI_Wtime();


    //
    // Task 1: Calculate the mean using all available processes.
    //
    MPI_Bcast(&localSize,1,MPI_FLOAT,0,MPI_COMM_WORLD); //Broadcast localSize
    MPI_Bcast(&globalSize,1,MPI_FLOAT,0,MPI_COMM_WORLD); //Broadcast globalSize

    // All ranks can now allocate memory for their local arrays.
    float *localData = (float*)malloc( localSize*sizeof(float));
    if( !localData )
  	{
  		printf( "Could not allocate memory for the local data array on rank %d.\n", rank );
  		MPI_Finalize();
  		return EXIT_FAILURE;
  	}

    // Collective version: Use MPI_Scatter.
  	MPI_Scatter(
  		globalData, localSize, MPI_FLOAT,						// Data being sent.
  		localData , localSize, MPI_FLOAT,						// Data being received.
  		0, MPI_COMM_WORLD									// Source rank and communicator - no tag!
  	);

    // Each process calculates their local mean.
    float local_sum = 0.0f;
    for( i=0; i<localSize; i++){
      local_sum += localData[i];
    }
    float local_mean = local_sum/globalSize;

    // Use MPI_Reduce() to get the answer back on root 0.
    float mean;
    MPI_Reduce(&local_mean, &mean, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    //
    // Task 2. Calculate the variance using all processes.
    //

    MPI_Bcast(&mean,1,MPI_FLOAT,0,MPI_COMM_WORLD);

    // Each process calculates their local variance.
    float local_sumSqrd = 0.0f;
    for( i=0; i<localSize; i++){
      local_sumSqrd += (localData[i]-mean)*(localData[i]-mean);
    }
    float local_variance = local_sumSqrd/globalSize;

    // Use Point-to-point communication arranged in Binary trees to get the answer back on root 0.
    //Second version in lec11
    int final = 1;
    int treeSize = numProcs;
    while(final && treeSize!=1){ //when nodes become 1, we are in the end

      if(rank>(treeSize/2-1)){ //the nodes after a half should send data
        MPI_Send(&local_variance,1,MPI_FLOAT,rank-treeSize/2,0,MPI_COMM_WORLD);
        final=0;
      }
      else{
        float recv_variance;
        MPI_Recv(&recv_variance,1,MPI_FLOAT,rank+treeSize/2,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        local_variance += recv_variance;
        treeSize = treeSize/2;

        if(treeSize==1)
          final =0;
      }
    }

    //
    // Output the results alongside a serial check.
    //
    if( rank==0 )
    {
        // Output the results of the timing now, before moving onto other calculations.
        printf( "Total time taken: %g s\n", MPI_Wtime() - startTime );

        // Your code MUST call this function after the mean and variance have been calculated using your parallel algorithms.
        // Do not modify the function itself (which is defined in 'cwk2_extra.h'), as it will be replaced with a different
        // version for the purpose of assessing. Also, don't just put the values from serial calculations here or you will lose marks.
        finalMeanAndVariance( mean, local_variance );
            // You should replace the first argument with your mean, and the second with your variance.

        // Check the answers against the serial calculations. This also demonstrates how to perform the calculations
        // in serial, which you may find useful. Note that small differences in floating point calculations between
        // equivalent parallel and serial codes are possible, as explained in Lecture 11.

        // Mean.
        float sum = 0.0;
        for( i=0; i<globalSize; i++ ) sum += globalData[i];
        float mean = sum / globalSize;

        // Variance.
        float sumSqrd = 0.0;
        for( i=0; i<globalSize; i++ ) sumSqrd += ( globalData[i]-mean )*( globalData[i]-mean );
        float variance = sumSqrd / globalSize;

        printf( "SERIAL CHECK: Mean=%g and Variance=%g.\n", mean, variance );

   }

    //
    // Free all resources (including any memory you have dynamically allocated), then quit.
    //
    if( rank==0 ) free( globalData );
    free( localData );

    MPI_Finalize();

    return EXIT_SUCCESS;
}
