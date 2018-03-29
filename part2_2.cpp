#include <immintrin.h>
#include <cmath>
#include <functional>
#include <chrono>
#include <random>
#include <iostream>
#include <cassert>
#include <omp.h>
using namespace std;
const int N = 16*1'000'000;
const int SAMPLE_NUM = 1000;
const int NUM_THREADS = 4;
const int NUM_DIMENSIONS = 24;
const int HIS_NUMS = 10;

int main () {
  alignas(32) static float points[N][NUM_DIMENSIONS];
  alignas(32) static long histgrams[NUM_DIMENSIONS][HIS_NUMS];
  alignas(32) static long count[NUM_DIMENSIONS];

  // alignas(32) static float a[N], b[N], c[N], d[N], e[N], f[N], g[N], h[N];

  std::default_random_engine eng;
  std::uniform_real_distribution<float> dist(-1, 1);

  double start = omp_get_wtime();
  omp_set_num_threads(NUM_THREADS);
  #pragma omp parallel
  {
  #pragma omp for
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < NUM_DIMENSIONS; ++j) {
        points[i][j] = dist(eng);
      }
    }
  }
  double end = omp_get_wtime();

  double start_compute = omp_get_wtime();
  for (int i = 0; i < N; ++i) {
    float distance_sq_sum = 0.0;
    for (int j = 0; j < NUM_DIMENSIONS; ++j) {
      ++ count[j];
      distance_sq_sum += points[i][j]*points[i][j];
      if (distance_sq_sum < 1) {
        // on j dimensions
        ++histgrams[j][(int)(sqrt(distance_sq_sum)*HIS_NUMS)-1];
      } else {
        break;
      }
    }
  }
  double end_compute = omp_get_wtime();

  printf("%f\n", end-start);
  printf("%f\n", end_compute - start_compute);

  for (int i = 0; i < NUM_DIMENSIONS; ++i) {
    for (int j = 0; j < HIS_NUMS; ++j) {
      printf("%ld ", histgrams[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  for (int i = 0; i < NUM_DIMENSIONS; ++i) {
    printf("count %d %ld\n",i,  count[i]);
  }

  return 0;
}
