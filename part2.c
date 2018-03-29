#include <immintrin.h>
#include <cmath>
#include <functional>
#include <chrono>
#include <random>
#include <iostream>
#include <cassert>

const int N = 16*1'000'000;
const int SAMPLE_NUM;

int main () {
  alignas(32) static float x[N], y[N], z[N], w[N];
  // alignas(32) static float u[N], v[N], p[N], q[N];

  std::default_random_engine eng;
  std::uniform_real_distribution<float> dist(-1, 1);
  for (int i = 0; i < N; i++) {
      x[i] = dist(eng);
      y[i] = dist(eng);
      z[i] = dist(eng);
      w[i] = dist(eng);

      // u[i] = dist(eng);
      // v[i] = dist(eng);
      // p[i] = dist(eng);
      // q[i] = dist(eng);
  }

  int count = 0;
  while (count < SAMPLE_NUM) {
    
    ++count;
  }


}
