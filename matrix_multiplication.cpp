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
void FillMatrix(double **matrix, int n);
void TransposeMatrix(double **matrix, int n);
void PrintMatrix(double **matrix, int n);
void FreeMatrix(double **matrix, int n);
double SerailMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n);
double ParallelMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n);
double ParallelImprovedMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n);
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

    for (int number_of_raws = 200; number_of_raws <= 600; number_of_raws += 200)
    { // call through this loop for each test case
        //std::cout << i << "\n";
        int runing_number_of_samples = intial_number_of_samples;
        run_program(number_of_raws, &finalized_serial_time, &finalized_parallel_time, &finalized_paralel_improved_time, &runing_number_of_samples, true);
        printf("No of samples runing for %d raws matrix = %d\n", number_of_raws, runing_number_of_samples);
        run_program(number_of_raws, &finalized_serial_time, &finalized_parallel_time, &finalized_paralel_improved_time, &runing_number_of_samples, false);
        printf("Time for serial multiplication of %d raws matrix = %f\n", number_of_raws, finalized_serial_time);
        printf("Time for Parallel multiplication of %d raws matrix = %f\n", number_of_raws, finalized_parallel_time);
        printf("Time for Parallel Improved multiplication of %d raws matrix = %f\n", number_of_raws, finalized_paralel_improved_time);
        std::cout << "\n";
    }
}

void run_program(int number_of_raws, double *finalized_serial_time, double *finalized_parallel_time, double *finalized_paralel_improved_time, int *number_of_samples, bool is_counting_no_of_samples)
{
    double *serial_time_array = new double[*number_of_samples];           //time array of serial multiplication
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
        //std::cout << serial_sample << "\n";
        //std::cout << parallel_sample << "\n";
        //std::cout << parallel_improved_sample << "\n";
        int max = serial_sample;
        if (max < parallel_sample)
        {
            max = parallel_sample;
        }
        if (max < parallel_improved_sample)
        {
            max = parallel_improved_sample;
        }
        if(max>100){
            max=100;
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

    FillMatrix(matrix_a, n);
    FillMatrix(matrix_b, n);
    //PrintMatrix(matrix_a,n);
    //std::cout << "\n";
    //PrintMatrix(matrix_b,n);
    //std::cout << "\n";

    *serial_time = SerailMultiply(matrix_a, matrix_b, matrix_c, n);
    *parallel_time = ParallelMultiply(matrix_a, matrix_b, matrix_c, n);
    //PrintMatrix(matrix_c,n);
    //std::cout << "\n";
    TransposeMatrix(matrix_b, n);
    //PrintMatrix(matrix_b,n);
    //std::cout << "\n";
    *paralel_improved_time = ParallelImprovedMultiply(matrix_a, matrix_b, matrix_c, n);
    //PrintMatrix(matrix_c,n);
    //std::cout << "\n";

    FreeMatrix(matrix_a, n);
    FreeMatrix(matrix_b, n);
    FreeMatrix(matrix_c, n);
}

double GetMean(double *array, int n)
{
    double total = 0.0;
    for (int i = 0; i < n; i++)
    {
        total = total + array[i];
    }
    return total / (n * 1.0);
}

double GetSD(double *array, double mean, int n)
{
    double standardDeviation = 0.0;
    for (int i = 0; i < n; ++i)
    {
        standardDeviation += pow(array[i] - mean, 2);
    }
    return sqrt(standardDeviation / n);
}

int GetSampleSize(double *array, int n)
{
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

void FillMatrix(double **matrix, int n)
{ // Random generated double values for matrix
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            matrix[i][j] = rand() / 1.01;
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
#pragma omp parallel
    {
        int i, j, k;
#pragma omp for
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                double multiply_sum = 0;
                for (k = 0; k < n; k++)
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
#pragma omp parallel
    {
        int i, j, k;
#pragma omp for
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                double multiply_sum = 0;
                for (k = 0; k < n; k++)
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
