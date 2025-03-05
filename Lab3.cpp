#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

using namespace std;

void simple_iteration_parallel_for(int N, vector<double>& a, const vector<double>& b, vector<double>& x, double tolerance, int max_iterations, int threads)
{
    double tau = 0.01;
    double b_mod = 0.0;
    for (int i = 0; i < N; ++i)
    {
        b_mod += b[i] * b[i];
    }
    b_mod = sqrt(b_mod);
    double prev_diff = __DBL_MAX__;

    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
        double x_mod = 0.0;
        for (int i = 0; i < N; ++i)
        {
            x_mod += (a[i] * x[i] - b[i]) * (a[i] * x[i] - b[i]);
        }
        x_mod = sqrt(x_mod);
        double diff = x_mod / b_mod;
        if (diff < tolerance)
        {
            printf("Break on iteration: %d\n", iteration);
            break;
        }
        if (diff > prev_diff)
            tau = -tau;
        else
            prev_diff = diff;
        //printf("Iteration: %d %f\n", iteration, diff);

        vector<double> x_next(N, 0.0);
#pragma omp parallel for num_threads(threads)
        for (int i = 0; i < N; ++i)
        {
            x_next[i] = x[i] - tau * (a[i] * x[i] - b[i]);
        }
        x = x_next;
    }
}

void simple_iteration_parallel(int N, vector<double>& a, const vector<double>& b, vector<double>& x, double tolerance, int max_iterations, int threads)
{
    double tau = 0.01;
    double b_mod = 0.0;
    for (int i = 0; i < N; ++i)
    {
        b_mod += b[i] * b[i];
    }
    b_mod = sqrt(b_mod);
    double prev_diff = __DBL_MAX__;

    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
        double x_mod = 0.0;
        for (int i = 0; i < N; ++i)
        {
            x_mod += (a[i] * x[i] - b[i]) * (a[i] * x[i] - b[i]);
        }
        x_mod = sqrt(x_mod);
        double diff = x_mod / b_mod;
        if (diff < tolerance)
        {
            printf("Break on iteration: %d\n", iteration);
            break;
        }
        if (diff > prev_diff)
            tau = -tau;
        else
            prev_diff = diff;
        //printf("Iteration: %d %f\n", iteration, diff);

        vector<double> x_next(N, 0.0);
#pragma omp parallel num_threads(threads)
        {
            for (int i = 0; i < N; ++i)
            {
                x_next[i] = x[i] - tau * (a[i] * x[i] - b[i]);
            }
        }
        x = x_next;
    }
}

int main() {
    int N = 10000; // Размерность системы
    double tolerance = 1e-10;
    int max_iterations = 10000;

    // Вектор b
    vector<double> B(N, N + 1);

    // Вариант 1
    // Диагональ А
    vector<double> A1(N, 2.0);
    // Решение
    vector<double> X1(N, 1.0);
    auto start1 = chrono::high_resolution_clock::now();
    simple_iteration_parallel_for(N, A1, B, X1, tolerance, max_iterations, 4);
    auto end1 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed1 = end1 - start1;
    //for (int i = 0; i < N; i++)
    //{
    //    printf("%f ", X1[i]);
    //}
    //printf("\n");

    // Вариант 2
    // Диагональ А
    vector<double> A2(N, 2.0);
    // Решение
    vector<double> X2(N, 1.0);
    auto start2 = chrono::high_resolution_clock::now();
    simple_iteration_parallel(N, A2, B, X2, tolerance, max_iterations, 4);
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed2 = end2 - start2;

    cout << "Время выполнения варианта 1: " << elapsed1.count() << " секунд" << endl;
    cout << "Время выполнения варианта 2: " << elapsed2.count() << " секунд" << endl;

    return 0;
}
