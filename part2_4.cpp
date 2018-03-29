#include <immintrin.h>
#include <cmath>
#include <functional>
#include <chrono>
#include <random>
#include <iostream>
#include <cassert>
#include <omp.h>
using namespace std;

#define NUM_DIMENSIONS 16
#define HIS_NUMS 100
#define NUM_THREADS 4
const int N = 15;

int main() {

  uniform_real_distribution<float> dist(-1, 1);
  static int histgram[NUM_DIMENSIONS][HIS_NUMS+1];
  static int num_sampling[NUM_DIMENSIONS];

  omp_set_num_threads(NUM_THREADS);
  double start = omp_get_wtime();
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    default_random_engine eng(thread_id);
  #pragma omp for
    for (int i = 0; i < N; ++i ) {
      float distance_sq_sum = 2.0;
      while (distance_sq_sum > 1.0) {
        distance_sq_sum = 0.0;
        for (int j = 0; j < NUM_DIMENSIONS; ++j) {
          float ran_num = dist(eng);
          distance_sq_sum += ran_num*ran_num;
          if (distance_sq_sum <= 1.0 && num_sampling[j] < N) {
            ++num_sampling[j];
            ++histgram[j][(int)(sqrt(distance_sq_sum)*HIS_NUMS)];
          }
          if (distance_sq_sum > 1.0) break;
        }
      }
    }
  }
  double end = omp_get_wtime();
  printf("using time: %f for %d threads.\n", end-start, NUM_THREADS);

  for (int i = 0; i < NUM_DIMENSIONS; ++i) {
    for (int j = 0; j < HIS_NUMS; ++j) {
      printf("%d", histgram[i][j]);
    }
    printf("\n");
  }
  return 0;
}
