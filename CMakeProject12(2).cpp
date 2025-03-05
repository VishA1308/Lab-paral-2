#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

const double PI = 3.14159265358979323846;
const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

void print_system_info() {
    
     // Print CPU information
     system("lscpu");

     // Print server name
     system("cat /sys/devices/virtual/dmi/id/product_name");
 
     // Print NUMA nodes information
     system("numactl --hardware");
}

// Функция для вычисления значения интегрируемой функции
double func(double x) {
    return exp(-x * x);
}

// Функция для измерения времени
double wtime() {
    return omp_get_wtime();
}

// Параллельная версия интегрирования с использованием OpenMP и атомарных операций
double integrate_omp(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel
    {
        double local_sum = 0.0; // Локальная переменная для каждого потока
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++) {
            local_sum += func(a + h * (i + 0.5));
        }

        // Обновляем общую сумму атомарно
#pragma omp atomic
        sum += local_sum;
    }

    sum *= h;
    return sum;
}

double integrate(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));
    sum *= h;
    return sum;
}

double run_serial() {
    double t = wtime();
    double res = integrate(func, a, b, nsteps);
    t = wtime() - t;
    printf("Result (serial): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

double run_parallel(int num_threads) {
    omp_set_num_threads(num_threads); // Установка количества потоков
    double t = wtime();
    double res = integrate_omp(func, a, b, nsteps);
    t = wtime() - t;
    printf("Result (parallel): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

int main(int argc, char** argv) {
    print_system_info();
    printf("Integration f(x) on [-4.0, 4.0], nsteps = %d\n", nsteps);

    double tserial = run_serial();

    int thread_counts[] = { 1, 2, 4, 7, 8, 16, 20, 40 };
    for (int i = 0; i < sizeof(thread_counts) / sizeof(thread_counts[0]); i++) {
        int num_threads = thread_counts[i];
        double tparallel = run_parallel(num_threads);
        printf("Execution time (parallel, %d threads): %.6f\n", num_threads, tparallel);
        printf("Speedup: %.2f\n", tserial / tparallel);
    }

    return 0;
}
