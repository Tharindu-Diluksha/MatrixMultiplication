#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <chrono>
#include <omp.h>

struct timeval time_for_srand;
struct timespec begin, end;
double elapsed;

void initialize(int n);
void CreateFullMatrix(double **matrix, int n);
void FillMatrix(double **matrix, int n);
void TransposeMatrix(double **matrix, int n);
void PrintMatrix(double **matrix, int n);
void FreeMatrix(double **matrix, int n);
void SerailMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n);
void ParallelMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n);
void ParallelImprovedMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n);
void StartTime();
void StopTime();
double GetTime();

/*
    compile => g++ -std=c++11 matrix_multiplication.cpp -o matrix
    run=> ./matrix
*/

main(int argc, char *argv[])
{
    //int n = 800;
    

    for (int i = 200; i <= 2000; i += 200)
    { // call through this loop for each test case
        //std::cout << i << "\n";
        srand((time_for_srand.tv_sec * 1000) + (time_for_srand.tv_usec / 1000));
        initialize(i);
    }
    
}

void initialize(int n){
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
    
    SerailMultiply(matrix_a, matrix_b, matrix_c, n);
    ParallelMultiply(matrix_a, matrix_b, matrix_c, n);
    //PrintMatrix(matrix_c,n);
    //std::cout << "\n";
    TransposeMatrix(matrix_b, n);
    //PrintMatrix(matrix_b,n);
    //std::cout << "\n";
    ParallelImprovedMultiply(matrix_a, matrix_b, matrix_c, n);
    //PrintMatrix(matrix_c,n);
    //std::cout << "\n";

    FreeMatrix(matrix_a, n);
    FreeMatrix(matrix_b, n);
    FreeMatrix(matrix_c, n);
    std::cout << "\n";
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
    
    FreeMatrix(matrix_t,n);
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
void SerailMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n)
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
    printf("Serial Time for %d number of columns and rows = %f \n", n, GetTime());
}

// Parallel Multiplication of two matrices
void ParallelMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n)
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
    printf("Parallel Time for %d number of columns and rows = %f \n", n, GetTime());
}

// Parallel Improved Multiplication of two matrices
void ParallelImprovedMultiply(double **matrix_a, double **matrix_b, double **matrix_c, int n)
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
                    multiply_sum += matrix_a[i][k] * matrix_b[j][k];
                }
                matrix_c[i][j] = multiply_sum;
            }
        }
    }
    StopTime();
    printf("Parallel Improved Time for %d number of columns and rows = %f \n", n, GetTime());
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
