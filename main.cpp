#include <immintrin.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>

using namespace std;

FILE *dummyfile;
int repeat = 10000000;


typedef double vec1d[1];
typedef double vec3d[3];
typedef double vec4d[4];
typedef double mat33d[3][3];
typedef double mat34d[3][4];

__m256i mask0111;
__m256i gatherMask036;
__m256i gatherMask147;
__m256i gatherMask258;


bool comp1(vec1d a, vec1d b) {
  return a[0] == b[0];
}

bool comp(vec3d a, vec3d b) {
  return a[0] == b[0] && 
         a[1] == b[1] &&
         a[2] == b[2];
}

bool comp(mat33d a, mat33d b) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (a[i][j] != b[i][j]) 
        return false;
    }
  }
  return true;
}

bool comp(mat34d a, mat34d b) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (a[i][j] != b[i][j]) 
        return false;
    }
  }
  return true;
}

void rand(double &c) {
  c = (double) rand() / RAND_MAX;
}

void rand1(vec1d c) {
  c[0] = (double) rand() / RAND_MAX;
}

void rand(vec3d c) {
  for (int i = 0; i < 3; i++) {
    rand(c[i]);
  }
}

void rand(mat33d c) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      rand(c[i][j]);
    }
  }
}

void rand(mat34d c) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      rand(c[i][j]);
    }
  }
}

template <typename inA, typename inB, typename out>
bool verify(void(*f)(inA, inB, out), void(*g)(inA, inB, out), bool(*comp)(out, out), void(*randA)(inA), void(*randB)(inB)) {
  inA a;
  inB b;
  randA(a);
  randB(b);
  
  double temp1[sizeof(out) / sizeof(double) + 4];
  double temp2[sizeof(out) / sizeof(double) + 4];

  out &c1 = *(out*) &temp1;
  f(a, b, c1);

  out &c2 = *(out*) &temp2;
  g(a, b, c2);
  bool result = comp(c1, c2);
  return result;
}

template <typename inA, typename inB, typename out>
clock_t _benchmark(void(*f)(inA, inB, out), void(*randA)(inA), void(*randB)(inB)) {
  inA *a = new inA[repeat + 1];
  inB *b = new inB[repeat + 1];
  out *c = new out[repeat + 1];
  for (int i = 0; i < repeat; i++) {
    randA(a[i]);
    randB(b[i]);
  }

  clock_t timer1, timer2;
  timer1 = clock();
  for (int i = 0; i < repeat; i++) {
    f(a[i], b[i], c[i]);
  }

  timer2 = clock();
  double dummy = *(double*)&(c[rand() % repeat]);
  fprintf(dummyfile, "%f", *(double*)&dummy);
  delete[] a;
  delete[] b;
  delete[] c;
  return timer2 - timer1;
}

#define benchmark(inA, inB, out, f, randA, randB) \
  cout << #f << ": " << _benchmark<inA, inB, out>(f, randA, randB) << endl

inline
void vec3adds(vec3d a, vec3d b, vec3d c) {
  for (int i = 0; i < 3; i++) {
    c[i] = a[i] + b[i];
  }
}

inline
void vec3addv(vec4d a, vec4d b, vec4d c) {
  __m256d tempa = _mm256_loadu_pd(a);
  __m256d tempb = _mm256_loadu_pd(b);
  __m256d tempc = _mm256_add_pd(tempa, tempb);
  _mm256_storeu_pd(c, tempc);
}

inline
void vec3addvmask(vec3d a, vec3d b, vec3d c) {
  _mm256_maskstore_pd(c, mask0111,
    _mm256_add_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b)));
}

inline
void vec3addvaligned(vec4d a, vec4d b, vec4d c) {
  __m256d tempa = _mm256_load_pd(a);
  __m256d tempb = _mm256_load_pd(b);
  __m256d tempc = _mm256_add_pd(tempa, tempb);
  _mm256_store_pd(c, tempc);
}

inline
void vec3dots(vec3d a, vec3d b, vec1d c) {
  double temp = 0;
  for (int i = 0; i < 3; i++) {
    temp += a[i] * b[i];
  }
  c[0] = temp;
}

inline
void vec3dotv(vec3d a, vec3d b, vec1d c) {
  __m256d temp = _mm256_mul_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b));
  __m128d temp2 = _mm256_extractf128_pd(temp, 1);
  temp = _mm256_hadd_pd(temp, temp);
  temp2 = _mm_add_pd(temp2, _mm256_extractf128_pd(temp, 0));
  c[0] = _mm_cvtsd_f64(temp2);
}

inline
void vec3dotv2(vec3d a, vec3d b, vec1d c) {
  __m256d tempa = _mm256_maskload_pd(a, mask0111);
  __m256d tempb = _mm256_loadu_pd(b);
  __m256d ab = _mm256_mul_pd(tempa, tempb);
  __m256d temp2 = _mm256_hadd_pd(ab, ab);
  __m128d lo = _mm256_extractf128_pd(temp2, 0);
  __m128d hi = _mm256_extractf128_pd(temp2, 1);
  __m128d res = _mm_add_sd(hi, lo);
  _mm_store_sd(c, res);
}

inline
void vec3dotv3(vec3d a, vec3d b, vec1d c) {
  __m256d temp = _mm256_mul_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b));
  __m256d temp2 = _mm256_hadd_pd(temp, temp);
  __m128d hi = _mm256_extractf128_pd(temp, 1);
  __m128d lo = _mm256_extractf128_pd(temp2, 0);
  _mm_maskstore_pd(c, _mm_set_epi64x(0, ULLONG_MAX), _mm_add_sd(hi, lo));
}

inline
void vec3exts(vec3d a, vec3d b, mat33d c) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      c[i][j] = a[i] * b[j];
    }
  }
}

inline
void vec3extv(vec3d a, vec3d b, mat33d c) {
  __m256d temp = _mm256_loadu_pd(b);
  _mm256_storeu_pd(c[0], _mm256_mul_pd(_mm256_set1_pd(a[0]), temp));
  _mm256_storeu_pd(c[1], _mm256_mul_pd(_mm256_set1_pd(a[1]), temp));
  _mm256_storeu_pd(c[2], _mm256_mul_pd(_mm256_set1_pd(a[2]), temp));
}

inline
void mat33vec3s(mat33d a, vec3d b, vec3d c) {
  for (int i = 0; i < 3; i++) {
    double temp = 0;
    for (int j = 0; j < 3; j++) {
      temp += a[i][j] * b[j];
    }
    c[i] = temp;
  }
}

inline
void mat33vec3vdot(mat33d a, vec3d b, vec3d c) {
  __m256d tempb = _mm256_maskload_pd(b, mask0111);
  {
    __m256d temp0 = _mm256_mul_pd(tempb, _mm256_loadu_pd(a[0]));
    temp0 = _mm256_hadd_pd(temp0, temp0);
    __m128d hi = _mm256_extractf128_pd(temp0, 1);
    __m128d lo = _mm256_extractf128_pd(temp0, 0);
    _mm_store_sd(c, _mm_add_pd(hi, lo));
  } {
    __m256d temp0 = _mm256_mul_pd(tempb, _mm256_loadu_pd(a[1]));
    temp0 = _mm256_hadd_pd(temp0, temp0);
    __m128d hi = _mm256_extractf128_pd(temp0, 1);
    __m128d lo = _mm256_extractf128_pd(temp0, 0);
    _mm_store_sd(c + 1, _mm_add_pd(hi, lo));
  } {
    __m256d temp0 = _mm256_mul_pd(tempb, _mm256_loadu_pd(a[2]));
    temp0 = _mm256_hadd_pd(temp0, temp0);
    __m128d hi = _mm256_extractf128_pd(temp0, 1);
    __m128d lo = _mm256_extractf128_pd(temp0, 0);
    _mm_store_sd(c + 2, _mm_add_pd(hi, lo));
  }
}

inline
void mat33vec3vfused1(mat33d a, vec3d b, vec3d c) {
  __m256d tempb = _mm256_maskload_pd(b, mask0111);
  __m256d temp0 = _mm256_mul_pd(tempb, _mm256_loadu_pd(a[0]));
  __m256d temp1 = _mm256_mul_pd(tempb, _mm256_loadu_pd(a[1]));
  __m256d temp2 = _mm256_mul_pd(tempb, _mm256_loadu_pd(a[2]));
  __m256d temp3 = _mm256_hadd_pd(temp0, temp1);
  __m256d temp4 = _mm256_hadd_pd(temp2, temp2);
  __m256d temp5 = _mm256_permute2f128_pd(temp3, temp4, 0b00100000);
  __m256d temp6 = _mm256_permute2f128_pd(temp3, temp4, 0b00110001);
  __m256d res = _mm256_add_pd(temp5, temp6);
  _mm256_maskstore_pd(c, mask0111, res);
}

inline
void mat33vec3vfused2(mat33d a, vec3d b, vec3d c) {
  __m256d tempb = _mm256_loadu_pd(b);
  __m256d temp0 = _mm256_mul_pd(tempb, _mm256_maskload_pd(a[0], mask0111));
  __m256d temp1 = _mm256_mul_pd(tempb, _mm256_maskload_pd(a[1], mask0111));
  __m256d temp2 = _mm256_mul_pd(tempb, _mm256_maskload_pd(a[2], mask0111));
  __m256d temp3 = _mm256_permute2f128_pd(temp0, temp2, 0b00100001);
  __m256d temp4 = _mm256_permute2f128_pd(temp1, temp2, 0b00100001);
  temp4 = _mm256_permute_pd(temp4, 0b0101);
  temp0 = _mm256_hadd_pd(temp0, temp0);
  temp1 = _mm256_hadd_pd(temp1, temp1);
  temp2 = _mm256_blend_pd(temp2, temp0, 0b0001);
  temp2 = _mm256_blend_pd(temp2, temp1, 0b0010);
  temp2 = _mm256_add_pd(temp2, temp3);
  temp2 = _mm256_add_pd(temp2, temp4);
  _mm256_maskstore_pd(c, mask0111, temp2);
}

inline
void mat33vec3vbroadcast(mat33d _a, vec3d b, vec3d c) {

  double *a = (double*) _a;
  __m256d temp0 = _mm256_i64gather_pd(a, gatherMask036, 1);
  __m256d tempb0 = _mm256_set1_pd(b[0]);
  __m256d res0 = _mm256_mul_pd(temp0, tempb0);

  __m256d temp1 = _mm256_i64gather_pd(a, gatherMask147, 1);
  __m256d tempb1 = _mm256_set1_pd(b[1]);
  __m256d res1 = _mm256_mul_pd(temp1, tempb1);

  __m256d temp2 = _mm256_i64gather_pd(a, gatherMask258, 1);
  __m256d tempb2 = _mm256_set1_pd(b[2]);
  __m256d res2 = _mm256_mul_pd(temp2, tempb2);

  __m256d res = _mm256_add_pd(res0, _mm256_add_pd(res1, res2));
  _mm256_maskstore_pd(c, mask0111, res);
}


void init() {
  dummyfile = tmpfile();
  mask0111 = _mm256_set_epi64x(0, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX);
  gatherMask036 = _mm256_set_epi64x(0, 64, 32, 0);
  gatherMask147 = _mm256_set_epi64x(0, 72, 40, 8);
  gatherMask258 = _mm256_set_epi64x(0, 80, 48, 16);
}

int main (int argc, char **argv) {
  if (argc > 1) {
    repeat = atoi(argv[1]);
  }
  init();
  cout << "verify\n"
       << verify<vec4d, vec4d, vec4d>(vec3adds, vec3addv, comp, rand, rand)
       << verify<vec3d, vec3d, vec3d>(vec3adds, vec3addvmask, comp, rand, rand)
      // << verify<vec4d, vec4d, vec4d>(vec3adds, vec3addvaligned, comp, rand, rand)
       << verify<vec3d, vec3d, vec1d>(vec3dots, vec3dotv, comp1, rand, rand)
       << verify<vec3d, vec3d, vec1d>(vec3dots, vec3dotv2, comp1, rand, rand)
       << verify<vec3d, vec3d, vec1d>(vec3dots, vec3dotv3, comp1, rand, rand)
       << verify<vec3d, vec3d, mat33d>(vec3exts, vec3extv, comp, rand, rand)
       << verify<mat33d, vec3d, vec3d>(mat33vec3s, mat33vec3vdot, comp, rand, rand)
       << verify<mat33d, vec3d, vec3d>(mat33vec3s, mat33vec3vfused1, comp, rand, rand)
       << verify<mat33d, vec3d, vec3d>(mat33vec3s, mat33vec3vfused2, comp, rand, rand)
       << verify<mat33d, vec3d, vec3d>(mat33vec3s, mat33vec3vbroadcast, comp, rand, rand)
              ;

  cout << "\nbenchmark\n";
  benchmark(vec3d, vec3d, vec3d, vec3adds, rand, rand);
  benchmark(vec4d, vec4d, vec4d, vec3addv, rand, rand);
  benchmark(vec3d, vec3d, vec3d, vec3addvmask, rand, rand);
 // benchmark(vec4d, vec4d, vec4d, vec3addvaligned, rand, rand);
  cout << endl;
  benchmark(vec3d, vec3d, vec1d, vec3dots, rand, rand);
  benchmark(vec3d, vec3d, vec1d, vec3dotv, rand, rand);
  benchmark(vec3d, vec3d, vec1d, vec3dotv2, rand, rand);
  benchmark(vec3d, vec3d, vec1d, vec3dotv3, rand, rand);
  cout << endl;
  benchmark(vec3d, vec3d, mat33d, vec3exts, rand, rand);
  benchmark(vec3d, vec3d, mat33d, vec3extv, rand, rand);
  cout << endl;
  benchmark(mat33d, vec3d, vec3d, mat33vec3s, rand, rand);
  benchmark(mat33d, vec3d, vec3d, mat33vec3vdot, rand, rand);
  benchmark(mat33d, vec3d, vec3d, mat33vec3vfused1, rand, rand);
  benchmark(mat33d, vec3d, vec3d, mat33vec3vfused2, rand, rand);
  benchmark(mat33d, vec3d, vec3d, mat33vec3vbroadcast, rand, rand);


  fclose(dummyfile);
  return 0;
}
