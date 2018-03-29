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
const int N = 15;
const int num_hist_intervals = 10;
alignas(32) static float points[N][NUM_DIMENSIONS];

int main() {

  uniform_real_distribution<float> dist(-1, 1);
  static int count = 0;
  static int total = 0;
  double start = omp_get_wtime();
  static int histgram[HIS_NUMS];

  omp_set_num_threads(3);
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    default_random_engine eng(thread_id);
  #pragma omp for
    for (int i = 0; i < N; ++i ) {
      float distance_sq_sum = 2.0;
      float temp[NUM_DIMENSIONS];
      while (distance_sq_sum > 1.0) {
        distance_sq_sum = 0.0;
        for (int j = 0; j < NUM_DIMENSIONS; ++j) {
          temp[j] = dist(eng);
          distance_sq_sum += temp[j]*temp[j];
        }
        ++ total;
      }
      for (int k = 0; k < NUM_DIMENSIONS; ++k) {
        points[i][k] = temp[k];
      }
      ++ histgram[(int)(distance_sq_sum*HIS_NUMS)-1];
      ++ count;
      printf("thread_id: %d  count: %d rejected: %d\n",thread_id ,count, total-count);
    }

  }
  printf("rejected ratio: %f\n", (float)(total-count)/(float)(total));
  double end = omp_get_wtime();
  printf("sampling time: %f\n", end-start);

  // for (int i = 0; i < N; ++i) {
  //   for (int j = 0; j < NUM_DIMENSIONS; ++j) {
  //     printf("%d   %f", i,points[i][j]);
  //   }
  //   printf("\n");
  // }

  for (int i = 0; i < HIS_NUMS; ++i) {
    printf("%d  ", histgram[i]);
  }
  printf("\n");


  return 0;
}
