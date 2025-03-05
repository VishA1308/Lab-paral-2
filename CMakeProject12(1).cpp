#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/utsname.h>

// Function to get the current time in seconds
double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void matrix_vector_product_seq(double* a, double* b, double* c, int m, int n) {
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++) {
            c[i] += a[i * n + j] * b[j];
        }
    }
}

void matrix_vector_product_omp(double* a, double* b, double* c, int m, int n) {
#pragma omp parallel for
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++) {
            c[i] += a[i * n + j] * b[j];
        }
    }
}

void run_parallel(int m, int n) {
    double* a = (double*)malloc(sizeof(double) * m * n);
    double* b = (double*)malloc(sizeof(double) * n);
    double* c = (double*)malloc(sizeof(double) * m);

    if (!a || !b || !c) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize arrays in parallel
#pragma omp parallel
    {
        int threadid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? m : lb + items_per_thread;

        for (int i = lb; i < ub; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = i + j;
            }
            c[i] = 0.0;
        }
    }

    // Initialize vector b
#pragma omp parallel for
    for (int j = 0; j < n; j++) {
        b[j] = j;
    }

    // Perform matrix-vector multiplication
    matrix_vector_product_omp(a, b, c, m, n);

    // Free allocated memory
    free(a);
    free(b);
    free(c);
}

void run_sequential(int m, int n) {
    double* a = (double*)malloc(sizeof(double) * m * n);
    double* b = (double*)malloc(sizeof(double) * n);
    double* c = (double*)malloc(sizeof(double) * m);

    if (!a || !b || !c) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize arrays sequentially
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = i + j;
        }
        c[i] = 0.0;
    }

    // Initialize vector b sequentially
    for (int j = 0; j < n; j++) {
        b[j] = j;
    }

    // Perform matrix-vector multiplication sequentially
    matrix_vector_product_seq(a, b, c, m, n);

    // Free allocated memory
    free(a);
    free(b);
    free(c);
}

void print_system_info() {
    struct utsname sysinfo;
    uname(&sysinfo);

    printf("Operating System: %s %s %s\n", sysinfo.sysname, sysinfo.release, sysinfo.version);
     // Print CPU information
     system("lscpu");

     // Print server name
     system("cat /sys/devices/virtual/dmi/id/product_name");
 
     // Print NUMA nodes information
     system("numactl --hardware");
}

int main() {
    print_system_info();

    int sizes[] = {20000, 40000};
    int thread_counts[] = {1, 2, 4, 7, 8, 16, 20, 40};
    
    for (int s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int m = sizes[s];
        int n = sizes[s];

        // Measure sequential execution time
        double start_time_seq = cpuSecond();
        run_sequential(m, n);
        double end_time_seq = cpuSecond();
        
        printf("Size: %dx%d, Execution time (sequential): %f seconds\n", m, n, end_time_seq - start_time_seq);

        for (int t = 0; t < sizeof(thread_counts) / sizeof(thread_counts[0]); t++) {
            int num_threads = thread_counts[t];
            omp_set_num_threads(num_threads);

            // Measure execution time
            double start_time_par = cpuSecond();
            run_parallel(m, n);
            double end_time_par = cpuSecond();

            printf("Size: %dx%d, Threads: %d, Execution time (parallel): %f seconds\n", m, n, num_threads, end_time_par - start_time_par);
            printf("Speedup: %.2f\n", (end_time_seq - start_time_seq) / (end_time_par - start_time_par));
        }
    }

    return 0;
}
