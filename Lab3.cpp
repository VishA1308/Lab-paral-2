#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <cfloat>
using namespace std;

void simple_iteration_parallel_for(int N, vector<double>& a, const vector<double>& b, vector<double>& x, double tolerance, int max_iterations, int threads) {
    double tau = 0.01;
    double b_mod = 0.0;

    // Вычисление нормы вектора b
    for (int i = 0; i < N; ++i) {
        b_mod += b[i] * b[i];
    }
    b_mod = sqrt(b_mod);
    double prev_diff = DBL_MAX;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        double x_mod = 0.0;

        // Вычисление нормы текущего решения
        for (int i = 0; i < N; ++i) {
            x_mod += (a[i] * x[i] - b[i]) * (a[i] * x[i] - b[i]);
        }
        x_mod = sqrt(x_mod);
        double diff = x_mod / b_mod;

        if (diff < tolerance) {
            printf("Break on iteration: %d\n", iteration);
            break;
        }
        if (diff > prev_diff)
            tau = -tau;
        else
            prev_diff = diff;

        vector<double> x_next(N, 0.0);
#pragma omp parallel for num_threads(threads)
        for (int i = 0; i < N; ++i) {
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
    double prev_diff = 100000000;

    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
        vector<double> x_next(N, 0.0);
        double x_mod = 0.0;
#pragma omp parallel for reduction(+ : x_mod) num_threads(threads)
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

#pragma omp parallel num_threads(threads)
        {
            int thread = omp_get_thread_num();
            int start = thread * N / threads;
            int end = (thread + 1) * N / threads;
            //printf("iteration: %d, thread: %d, start: %d, end: %d\n", iteration, thread, start, end);
            for (int i = start; i < end; ++i)
            {
                x_next[i] = x[i] - tau * (a[i] * x[i] - b[i]);
            }
        }
        x = x_next;
    }
}

int main() {
    int N = 100000; // Размерность системы
    double tolerance = 1e-10;
    int max_iterations = 10000;

    // Вектор b
    vector<double> B(N, N + 1);

    // Массивы для хранения времени выполнения
    vector<double> times1(omp_get_max_threads());
    vector<double> times2(omp_get_max_threads());

    // Цикл по количеству потоков
    for (int threads = 1; threads <= 16; ++threads) {
        vector<double> A1(N, 2.0); // Диагональ A
        vector<double> X1(N, 1.0); // Начальное решение

        // Замер времени для варианта 1
        auto start1 = chrono::high_resolution_clock::now();
        simple_iteration_parallel_for(N, A1, B, X1, tolerance, max_iterations, threads);
        auto end1 = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed1 = end1 - start1;
        times1[threads - 1] = elapsed1.count();

        vector<double> A2(N, 2.0); // Диагональ A
        vector<double> X2(N, 1.0); // Начальное решение

        // Замер времени для варианта 2
        auto start2 = chrono::high_resolution_clock::now();
        simple_iteration_parallel(N, A2, B, X2, tolerance, max_iterations, threads);
        auto end2 = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed2 = end2 - start2;
        times2[threads - 1] = elapsed2.count();
    }

    // Вывод результатов
    cout << "Время выполнения варианта 1:" << endl;
    for (int i = 0; i < times1.size(); ++i) {
        cout << "Потоки: " << (i + 1) << " Время: " << times1[i] << " секунд" << endl;
    }

    cout << "Время выполнения варианта 2:" << endl;
    for (int i = 0; i < times2.size(); ++i) {
        cout << "Потоки: " << (i + 1) << " Время: " << times2[i] << " секунд" << endl;
    }

    return 0;
}
