#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <chrono>
#include <omp.h>
#include <cmath>

struct timeval time_for_srand;
struct timespec begin, end;
double elapsed;

void run_program(int number_of_raws, double *finalized_serial_time, double *finalized_parallel_time, double *finalized_paralel_improved_time, int *number_of_samples, bool is_counting_no_of_samples);
void initiate_run(int n, double *serial_time, double *parallel_time, double *paralel_improved_time);
void CreateFullMatrix(double **matrix, int n);
void FillMatrix(double **matrix, double *matrix_1D, int n);
void TransposeMatrix(double **matrix, int n);
void TransposeMatrix_1D(double *matrix, int n);
void PrintMatrix(double **matrix, int n);
void FreeMatrix(double **matrix, int n);
double SerailMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n);
double ParallelMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n);
double ParallelImprovedMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n);
double ParallelImprovedMultiply2(double *matrix_a, double *matrix_b, double *matrix_c, int n);
void StartTime();
void StopTime();
double GetTime();
double GetMean(double *array, int n);
double GetSD(double *array, double mean, int n);
int GetSampleSize(double *array, int n);

/*
    compile => g++ -fopenmp -std=c++11 matrix_multiplication.cpp -o matrix
    run=> ./matrix
*/

main(int argc, char *argv[])
{
    int intial_number_of_samples = 10;
    double finalized_serial_time = 0.0;
    double finalized_parallel_time = 0.0;
    double finalized_paralel_improved_time = 0.0;
    int runing_number_of_samples = intial_number_of_samples;

    for (int number_of_raws = 200; number_of_raws <= 2000; number_of_raws += 200)
    { // call through this loop for each test case
        int runing_number_of_samples = intial_number_of_samples;

        // Run for finding the sufficient number of samples //
        run_program(number_of_raws, &finalized_serial_time, &finalized_parallel_time, &finalized_paralel_improved_time, &runing_number_of_samples, true);
        printf("No of samples runing for %d rows matrix = %d\n", number_of_raws, runing_number_of_samples);

        //run for actual time measurements with the calculated number of samples //
        run_program(number_of_raws, &finalized_serial_time, &finalized_parallel_time, &finalized_paralel_improved_time, &runing_number_of_samples, false);

        printf("Time for sequential multiplication of %d rows matrix = %f\n", number_of_raws, finalized_serial_time);
        printf("Time for Parallel multiplication of %d rows matrix = %f\n", number_of_raws, finalized_parallel_time);
        printf("Time for Parallel Improved multiplication of %d rows matrix = %f\n", number_of_raws, finalized_paralel_improved_time);
        std::cout << "\n";
    }
}

void run_program(int number_of_raws, double *finalized_serial_time, double *finalized_parallel_time, double *finalized_paralel_improved_time, int *number_of_samples, bool is_counting_no_of_samples)
{   
    // This will run the matrix multiplication for both sample calculation and timemeasurements. If the value of the is_counting_no_of_samples is true then
    // it will calculate the no of sufficient samples and if it is false then it will calculate the actual time measurement(results).
    double *serial_time_array = new double[*number_of_samples];           //time array of sequential multiplication
    double *parallel_time_array = new double[*number_of_samples];         //time array of parallel multiplication
    double *paralel_improved_time_array = new double[*number_of_samples]; //time array of parallel improved multiplication
    srand((time_for_srand.tv_sec * 1000) + (time_for_srand.tv_usec / 1000));
    int sample_size = *number_of_samples;
    for (int sample_num = 0; sample_num < *number_of_samples; sample_num++)
    { // run program for given number of samples
        double serial_time = 0.0;
        double parallel_time = 0.0;
        double paralel_improved_time = 0.0;
        initiate_run(number_of_raws, &serial_time, &parallel_time, &paralel_improved_time); //initiate actual multiplication
        serial_time_array[sample_num] = serial_time;
        parallel_time_array[sample_num] = parallel_time;
        paralel_improved_time_array[sample_num] = paralel_improved_time;
    }

    if (is_counting_no_of_samples)
    { // Find the sufficient number of samples
        int serial_sample = GetSampleSize(serial_time_array, sample_size);
        int parallel_sample = GetSampleSize(parallel_time_array, sample_size);
        int parallel_improved_sample = GetSampleSize(paralel_improved_time_array, sample_size);
        int max = serial_sample;
        if (max < parallel_sample)
        {
            max = parallel_sample;
        }
        if (max < parallel_improved_sample)
        {
            max = parallel_improved_sample;
        }
        if (max > 100)
        {
            max = 100;
        }
        *number_of_samples = max;
    }

    else
    { // To run the multiplication without finding the no of sufficient samples
        *finalized_serial_time = GetMean(serial_time_array, sample_size);
        *finalized_parallel_time = GetMean(parallel_time_array, sample_size);
        *finalized_paralel_improved_time = GetMean(paralel_improved_time_array, sample_size);
    }
    delete[] serial_time_array;
    delete[] parallel_time_array;
    delete[] paralel_improved_time_array;
}

void initiate_run(int n, double *serial_time, double *parallel_time, double *paralel_improved_time)
{ //actually run the matrix multiplication
    double **matrix_a = new double *[n];
    double **matrix_b = new double *[n];
    double **matrix_c = new double *[n];
    CreateFullMatrix(matrix_a, n);
    CreateFullMatrix(matrix_b, n);
    CreateFullMatrix(matrix_c, n);

    double *matrix_a_1D = new double[n*n];
    double *matrix_b_1D = new double[n*n];
    double *matrix_c_1D = new double[n*n];

    FillMatrix(matrix_a, matrix_a_1D, n);
    FillMatrix(matrix_b, matrix_b_1D, n);

    *serial_time = SerailMultiply(matrix_a, matrix_b, matrix_c, n);     // call serail multiplication
    *parallel_time = ParallelMultiply(matrix_a, matrix_b, matrix_c, n); // call parallel multiplication

    // TransposeMatrix(matrix_b, n); // Taking transpose of B and put it in B.
    TransposeMatrix_1D(matrix_b_1D, n);
    // run the improved version of  parallel multiplication //
    //*paralel_improved_time = ParallelImprovedMultiply(matrix_a, matrix_b, matrix_c, n);// Only transposed matrix with parallel
    *paralel_improved_time = ParallelImprovedMultiply2(matrix_a_1D, matrix_b_1D, matrix_c_1D, n);

    FreeMatrix(matrix_a, n); // Free the memory locations of matrix
    FreeMatrix(matrix_b, n);
    FreeMatrix(matrix_c, n);
    delete[] matrix_a_1D;
    delete[] matrix_b_1D;
    delete[] matrix_c_1D;
}

double GetMean(double *array, int n)
{ //Calculate mean
    double total = 0.0;
    for (int i = 0; i < n; i++)
    {
        total = total + array[i];
    }
    return total / (n * 1.0);
}

double GetSD(double *array, double mean, int n)
{ //Calculate standard deviation
    double standardDeviation = 0.0;
    for (int i = 0; i < n; ++i)
    {
        standardDeviation += pow(array[i] - mean, 2);
    }
    return sqrt(standardDeviation / n);
}

int GetSampleSize(double *array, int n)
{ // Return the sufficient sample size
    double mean = GetMean(array, n);
    return pow(((100 * 1.960 * GetSD(array, mean, n)) / (5 * mean)), 2) + 1;
}

void CreateFullMatrix(double **matrix, int n)
{ // Create dynamic matrix according to the n value
    for (int i = 0; i < n; i++)
    {
        matrix[i] = new double[n];
    }
}

void FillMatrix(double **matrix, double *matrix_1D, int n)
{ // Random generated double values for matrix
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double element = rand() / 1.01;
            matrix[i][j] = element;
            matrix_1D[n*i + j] = element;
        }
    }
}

void TransposeMatrix(double **matrix, int n)
{ //Get the transpose matrix
    double **matrix_t = new double *[n];
    CreateFullMatrix(matrix_t, n);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            matrix_t[i][j] = matrix[i][j];
        }
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            matrix[i][j] = matrix_t[j][i];
        }
    }

    FreeMatrix(matrix_t, n);
}

void TransposeMatrix_1D(double *matrix, int n)
{ //Get the transpose matrix
    double *matrix_t = new double[n*n];
    // CreateFullMatrix(matrix_t, n);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            matrix_t[i*n+j] = matrix[i*n+j];
        }
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            matrix[i*n+j] = matrix_t[j*n+i];
        }
    }

    delete[] matrix_t;
}

void PrintMatrix(double **matrix, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void FreeMatrix(double **matrix, int n)
{ // Delete matrix memory
    for (int i = 0; i < n; ++i)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// Serail Multiplication of two matrices
double SerailMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n)
{
    StartTime();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double multiply_sum = 0;
            for (int k = 0; k < n; k++)
            {
                multiply_sum += matrix_a[i][k] * matrix_b[k][j];
            }
            matrix_c[i][j] = multiply_sum;
        }
    }
    StopTime();
    double exc_time = GetTime();
    //printf("Serial Time for %d number of columns and rows = %f \n", n, exc_time);
    return exc_time;
}

// Parallel Multiplication of two matrices
double ParallelMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n)
{
    StartTime();
    // #pragma omp parallel
    {
// int i, j, k;
// #pragma omp for
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            // #pragma omp parallel for
            for (int j = 0; j < n; j++)
            {
                double multiply_sum = 0;
                for (int k = 0; k < n; k++)
                {
                    multiply_sum += matrix_a[i][k] * matrix_b[k][j];
                }
                matrix_c[i][j] = multiply_sum;
            }
        }
    }
    StopTime();
    double exc_time = GetTime();
    //printf("Parallel Time for %d number of columns and rows = %f \n", n, exc_time);
    return exc_time;
}

// Parallel Improved Multiplication of two matrices
double ParallelImprovedMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n)
{ //Here the matix_b is transposed one
    StartTime();
    // #pragma omp parallel
    {
// int i, j, k;
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            // #pragma omp parallel for
            for (int j = 0; j < n; j++)
            {
                double multiply_sum = 0;
                for (int k = 0; k < n; k++)
                {
                    multiply_sum += matrix_a[i][k] * matrix_b[j][k];
                }
                matrix_c[i][j] = multiply_sum;
            }
        }
    }
    StopTime();
    double exc_time = GetTime();
    //printf("Parallel Improved Time for %d number of columns and rows = %f \n", n, exc_time);
    return exc_time;
}

// Parallel Improved Multiplication of two matrices - 2
double ParallelImprovedMultiply2(double *matrix_a, double *matrix_b, double *matrix_c, int n)
{
    StartTime();
// #pragma omp parallel
    {
        // int i, j, k;
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j=j+8)
            {
                double multiply_sum[] = {0,0,0,0,0,0,0,0};
                for (int k = 0; k < n; k++)
                {
                    multiply_sum[0] += matrix_a[i*n+k] * matrix_b[j*n+k];
                    // multiply_sum[0] += matrix_a[i][k+1] * matrix_b[j][k+1];
                    // multiply_sum[0] += matrix_a[i][k+2] * matrix_b[j][k+2];
                    // multiply_sum[0] += matrix_a[i][k+3] * matrix_b[j][k+3];
                    // multiply_sum[0] += matrix_a[i][k+4] * matrix_b[j][k+4];
                    // multiply_sum[0] += matrix_a[i][k+5] * matrix_b[j][k+5];
                    // multiply_sum[0] += matrix_a[i][k+6] * matrix_b[j][k+6];
                    // multiply_sum[0] += matrix_a[i][k+7] * matrix_b[j][k+7];

                    multiply_sum[1] += matrix_a[i*n+k] * matrix_b[(j+1)*n+k];
                    // multiply_sum[1] += matrix_a[i][k+1] * matrix_b[j][k+1];
                    // multiply_sum[1] += matrix_a[i][k+2] * matrix_b[j][k+2];
                    // multiply_sum[1] += matrix_a[i][k+3] * matrix_b[j][k+3];
                    // multiply_sum[1] += matrix_a[i][k+4] * matrix_b[j][k+4];
                    // multiply_sum[1] += matrix_a[i][k+5] * matrix_b[j][k+5];
                    // multiply_sum[1] += matrix_a[i][k+6] * matrix_b[j][k+6];
                    // multiply_sum[1] += matrix_a[i][k+7] * matrix_b[j][k+7];

                    multiply_sum[2] += matrix_a[i*n+k] * matrix_b[(j+2)*n+k];
                    // multiply_sum[2] += matrix_a[i][k+1] * matrix_b[j][k+1];
                    // multiply_sum[2] += matrix_a[i][k+2] * matrix_b[j][k+2];
                    // multiply_sum[2] += matrix_a[i][k+3] * matrix_b[j][k+3];
                    // multiply_sum[2] += matrix_a[i][k+4] * matrix_b[j][k+4];
                    // multiply_sum[2] += matrix_a[i][k+5] * matrix_b[j][k+5];
                    // multiply_sum[2] += matrix_a[i][k+6] * matrix_b[j][k+6];
                    // multiply_sum[2] += matrix_a[i][k+7] * matrix_b[j][k+7];

                    multiply_sum[3] += matrix_a[i*n+k] * matrix_b[(j+3)*n+k];
                    // multiply_sum[3] += matrix_a[i][k+1] * matrix_b[j][k+1];
                    // multiply_sum[3] += matrix_a[i][k+2] * matrix_b[j][k+2];
                    // multiply_sum[3] += matrix_a[i][k+3] * matrix_b[j][k+3];
                    // multiply_sum[3] += matrix_a[i][k+4] * matrix_b[j][k+4];
                    // multiply_sum[3] += matrix_a[i][k+5] * matrix_b[j][k+5];
                    // multiply_sum[3] += matrix_a[i][k+6] * matrix_b[j][k+6];
                    // multiply_sum[3] += matrix_a[i][k+7] * matrix_b[j][k+7];

                    multiply_sum[4] += matrix_a[i*n+k] * matrix_b[(j+4)*n+k];
                    // multiply_sum[4] += matrix_a[i][k+1] * matrix_b[j][k+1];
                    // multiply_sum[4] += matrix_a[i][k+2] * matrix_b[j][k+2];
                    // multiply_sum[4] += matrix_a[i][k+3] * matrix_b[j][k+3];
                    // multiply_sum[4] += matrix_a[i][k+4] * matrix_b[j][k+4];
                    // multiply_sum[4] += matrix_a[i][k+5] * matrix_b[j][k+5];
                    // multiply_sum[4] += matrix_a[i][k+6] * matrix_b[j][k+6];
                    // multiply_sum[4] += matrix_a[i][k+7] * matrix_b[j][k+7];
                    
                    multiply_sum[5] += matrix_a[i*n+k] * matrix_b[(j+5)*n+k];
                    // multiply_sum[5] += matrix_a[i][k+1] * matrix_b[j][k+1];
                    // multiply_sum[5] += matrix_a[i][k+2] * matrix_b[j][k+2];
                    // multiply_sum[5] += matrix_a[i][k+3] * matrix_b[j][k+3];
                    // multiply_sum[5] += matrix_a[i][k+4] * matrix_b[j][k+4];
                    // multiply_sum[5] += matrix_a[i][k+5] * matrix_b[j][k+5];
                    // multiply_sum[5] += matrix_a[i][k+6] * matrix_b[j][k+6];
                    // multiply_sum[5] += matrix_a[i][k+7] * matrix_b[j][k+7];

                    multiply_sum[6] += matrix_a[i*n+k] * matrix_b[(j+6)*n+k];
                    // multiply_sum[6] += matrix_a[i][k+1] * matrix_b[j][k+1];
                    // multiply_sum[6] += matrix_a[i][k+2] * matrix_b[j][k+2];
                    // multiply_sum[6] += matrix_a[i][k+3] * matrix_b[j][k+3];
                    // multiply_sum[6] += matrix_a[i][k+4] * matrix_b[j][k+4];
                    // multiply_sum[6] += matrix_a[i][k+5] * matrix_b[j][k+5];
                    // multiply_sum[6] += matrix_a[i][k+6] * matrix_b[j][k+6];
                    // multiply_sum[6] += matrix_a[i][k+7] * matrix_b[j][k+7];

                    multiply_sum[7] += matrix_a[i*n+k] * matrix_b[(j+7)*n+k];
                    // multiply_sum[7] += matrix_a[i][k+1] * matrix_b[j][k+1];
                    // multiply_sum[7] += matrix_a[i][k+2] * matrix_b[j][k+2];
                    // multiply_sum[7] += matrix_a[i][k+3] * matrix_b[j][k+3];
                    // multiply_sum[7] += matrix_a[i][k+4] * matrix_b[j][k+4];
                    // multiply_sum[7] += matrix_a[i][k+5] * matrix_b[j][k+5];
                    // multiply_sum[7] += matrix_a[i][k+6] * matrix_b[j][k+6];
                    // multiply_sum[7] += matrix_a[i][k+7] * matrix_b[j][k+7];

                }
                matrix_c[i*n+j] = multiply_sum[0];
                matrix_c[i*n+(j+1)] = multiply_sum[1];
                matrix_c[i*n+(j+2)] = multiply_sum[2];
                matrix_c[i*n+(j+3)] = multiply_sum[3];
                matrix_c[i*n+(j+4)] = multiply_sum[4];
                matrix_c[i*n+(j+5)] = multiply_sum[5];
                matrix_c[i*n+(j+6)] = multiply_sum[6];
                matrix_c[i*n+(j+7)] = multiply_sum[7];
            }
        }
    }
    StopTime();
    double exc_time = GetTime();
    // printf("Parallel Improved Time for %d number of columns and rows = %f \n", n, GetTime());    
    return exc_time;
}

void StartTime()
{ //Set the start time
    clock_gettime(CLOCK_MONOTONIC, &begin);
}

void StopTime()
{ //Set the end time
    clock_gettime(CLOCK_MONOTONIC, &end);
}

double GetTime()
{ // Get the time difference between start and end time
    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1000000000.0; // adding nano seconds values.
    return elapsed;
}
