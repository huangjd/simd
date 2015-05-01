#include <immintrin.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>

using namespace std;

FILE *dummyfile;
int ncount = 100;
int repeat = 100000;


typedef double vec1d[1];
typedef double vec3d[4];
//typedef double vec4d[4];
typedef vec3d vec4d;
//typedef double mat33d[3*3+3];
typedef double mat33d[12];
typedef double mat34d[3*3+4];

__m256i mask0111;
__m256i gatherMask036;
__m256i gatherMask147;
__m256i gatherMask258;


bool comp1(vec1d a, vec1d b) {
  return a[0] == b[0];
}

template <size_t n>
bool comp(double a[n], double b[n]) {
  for (size_t i = 0; i < n; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

bool comp(mat34d a, mat34d b) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (a[i*3+j] != b[i*3+j])
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

template <size_t n>
void rand(double c[n]) {
  for (size_t i = 0; i < n; i++) {
    rand(c[i]);
  }
}


void rand(mat34d c) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      rand(c[i*3+j]);
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
  inA *a = (inA*) aligned_alloc(64, (ncount + 1) * sizeof(inA));
  inB *b = (inB*) aligned_alloc(64, (ncount + 1) * sizeof(inB));
  out *c = (out*) aligned_alloc(64, (ncount + 1) * sizeof(out));
  for (int i = 0; i < ncount; i++) {
    randA(a[i]);
    randB(b[i]);
  }

  clock_t timer1, timer2;
  timer1 = clock();
  for (int j = 0; j < repeat; j++) {
    for (int i = 0; i < ncount; i++) {
      f(a[i], b[i], c[i]);
    }
  }

  timer2 = clock();
  double dummy = *(double*)&(c[rand() % ncount]);
  fprintf(dummyfile, "%f", *(double*)&dummy);
  free(a);
  free(b);
  free(c);
  return timer2 - timer1;
}

#define benchmark(inA, inB, out, f, randA, randB) \
  cout << #f << ": " << _benchmark<inA, inB, out>(f, randA, randB) << endl

//static inline void vec3adds(vec3d a, vec3d b, vec3d c) __attribute__((optimize("-O0")));

static inline
void vec3adds(vec3d a, vec3d b, vec3d c) {
  for (int i = 0; i < 3; i++) {
    c[i] = a[i] + b[i];
  }
}

//static inline void vec3addsunroll(vec3d a, vec3d b, vec3d c) __attribute__((optimize("-O0")));

static inline
void vec3addsunroll(vec3d a, vec3d b, vec3d c) {
  c[0] = a[0] + b[0];
  c[1] = a[1] + b[1];
  c[2] = a[2] + b[2];
}

static inline
void vec3addv(vec4d a, vec4d b, vec4d c) {
  __m256d tempa = _mm256_loadu_pd(a);
  __m256d tempb = _mm256_loadu_pd(b);
  __m256d tempc = _mm256_add_pd(tempa, tempb);
  _mm256_storeu_pd(c, tempc);
}

static inline
void vec3addvmask(vec3d a, vec3d b, vec3d c) {
  _mm256_maskstore_pd(c, mask0111,
    _mm256_add_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b)));
}

static inline
void vec3addvaligned(vec4d a, vec4d b, vec4d c) {
  __m256d tempa = _mm256_load_pd(a);
  __m256d tempb = _mm256_load_pd(b);
  __m256d tempc = _mm256_add_pd(tempa, tempb);
  _mm256_store_pd(c, tempc);
}

//static inline void vec3dots(vec3d a, vec3d b, vec1d c) __attribute__((optimize("-O0")));

static inline
void vec3dots(vec3d a, vec3d b, vec1d c) {
  double temp = 0;
  for (int i = 0; i < 3; i++) {
    temp += a[i] * b[i];
  }
  c[0] = temp;
}

//static inline void vec3dotsunroll(vec3d a, vec3d b, vec1d c) __attribute__((optimize("-O0")));

static inline
void vec3dotsunroll(vec3d a, vec3d b, vec1d c) {
  c[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline
void vec3dotv(vec3d a, vec3d b, vec1d c) {
  __m256d temp = _mm256_mul_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b));
  __m128d temp2 = _mm256_extractf128_pd(temp, 1);
  temp = _mm256_hadd_pd(temp, temp);
  temp2 = _mm_add_pd(temp2, _mm256_extractf128_pd(temp, 0));
  c[0] = _mm_cvtsd_f64(temp2);
}

static inline
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

static inline
void vec3dotv3(vec3d a, vec3d b, vec1d c) {
  __m256d temp = _mm256_mul_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b));
  __m256d temp2 = _mm256_hadd_pd(temp, temp);
  __m128d hi = _mm256_extractf128_pd(temp, 1);
  __m128d lo = _mm256_extractf128_pd(temp2, 0);
  _mm_maskstore_pd(c, _mm_set_epi64x(0, ULLONG_MAX), _mm_add_sd(hi, lo));
}

//static inline void vec3exts(vec3d a, vec3d b, mat33d c) __attribute__((optimize("-O0")));

static inline
void vec3exts(vec3d a, vec3d b, mat33d c) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      c[i*3+j] = a[i] * b[j];
    }
  }
}

//static inline void vec3extsunroll(vec3d a, vec3d b, mat33d c) __attribute__((optimize("-O0")));

inline
void vec3extsunroll(vec3d a, vec3d b, mat33d c) {
  c[0*3+0] = a[0] * b[0];
  c[0*3+1] = a[0] * b[1];
  c[0*3+2] = a[0] * b[2];
  c[1*3+0] = a[1] * b[0];
  c[1*3+1] = a[1] * b[1];
  c[1*3+2] = a[1] * b[2];
  c[2*3+0] = a[2] * b[0];
  c[2*3+1] = a[2] * b[1];
  c[2*3+2] = a[2] * b[2];
}

static inline
void vec3extv(vec3d a, vec3d b, mat33d c) {
  __m256d temp = _mm256_loadu_pd(b);
  _mm256_storeu_pd(c, _mm256_mul_pd(_mm256_set1_pd(a[0]), temp));
  _mm256_storeu_pd(c+3, _mm256_mul_pd(_mm256_set1_pd(a[1]), temp));
  _mm256_storeu_pd(c+6, _mm256_mul_pd(_mm256_set1_pd(a[2]), temp));
}

//static inline void mat33vec3s(mat33d a, vec3d b, vec3d c) __attribute__((optimize("-O0")));

static inline
void mat33vec3s(mat33d a, vec3d b, vec3d c) {
  for (int i = 0; i < 3; i++) {
    double temp = 0;
    for (int j = 0; j < 3; j++) {
      temp += a[i*3+j] * b[j];
    }
    c[i] = temp;
  }
}

//static inline void mat33vec3sunroll(mat33d a, vec3d b, vec3d c) __attribute__((optimize("-O0")));

static inline
void mat33vec3sunroll(mat33d a, vec3d b, vec3d c) {
  c[0] = a[0*3+0] * b[0] + a[0*3+1] * b[1] + a[0*3+2] * b[2];
  c[1] = a[1*3+0] * b[0] + a[1*3+1] * b[1] + a[1*3+2] * b[2];
  c[2] = a[2*3+0] * b[0] + a[2*3+1] * b[1] + a[2*3+2] * b[2];
}

static inline
void mat33vec3vdot(mat33d a, vec3d b, vec3d c) {
  __m256d tempb = _mm256_maskload_pd(b, mask0111);
  {
    __m256d temp0 = _mm256_mul_pd(tempb, _mm256_loadu_pd(a));
    temp0 = _mm256_hadd_pd(temp0, temp0);
    __m128d hi = _mm256_extractf128_pd(temp0, 1);
    __m128d lo = _mm256_extractf128_pd(temp0, 0);
    _mm_store_sd(c, _mm_add_pd(hi, lo));
  } {
    __m256d temp0 = _mm256_mul_pd(tempb, _mm256_loadu_pd(a+3));
    temp0 = _mm256_hadd_pd(temp0, temp0);
    __m128d hi = _mm256_extractf128_pd(temp0, 1);
    __m128d lo = _mm256_extractf128_pd(temp0, 0);
    _mm_store_sd(c + 1, _mm_add_pd(hi, lo));
  } {
    __m256d temp0 = _mm256_mul_pd(tempb, _mm256_loadu_pd(a+6));
    temp0 = _mm256_hadd_pd(temp0, temp0);
    __m128d hi = _mm256_extractf128_pd(temp0, 1);
    __m128d lo = _mm256_extractf128_pd(temp0, 0);
    _mm_store_sd(c + 2, _mm_add_pd(hi, lo));
  }
}

static inline
void mat33vec3vfused1(mat33d a, vec3d b, vec3d c) {
  __m256d tempb = _mm256_maskload_pd(b, mask0111);
  __m256d temp0 = _mm256_mul_pd(tempb, _mm256_loadu_pd(a));
  __m256d temp1 = _mm256_mul_pd(tempb, _mm256_loadu_pd(a+3));
  __m256d temp2 = _mm256_mul_pd(tempb, _mm256_loadu_pd(a+6));
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
  __m256d temp0 = _mm256_mul_pd(tempb, _mm256_maskload_pd(a, mask0111));
  __m256d temp1 = _mm256_mul_pd(tempb, _mm256_maskload_pd(a+3, mask0111));
  __m256d temp2 = _mm256_mul_pd(tempb, _mm256_maskload_pd(a+6, mask0111));
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

//static inline void mat33dmuls(mat33d a, mat33d b, mat33d c) __attribute__((optimize("-O0")));

static inline
void mat33dmuls(mat33d a, mat33d b, mat33d c) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double temp = 0;
      for (int k = 0; k < 3; k++) {
        temp += a[i*3+k] * b[k*3+j];
      }
      c[i*3+j] = temp;
    }
  }
}

//static inline void mat33dmulsunroll(mat33d a, mat33d b, mat33d c) __attribute__((optimize("-O0")));

static inline
void mat33dmulsunroll(mat33d a, mat33d b, mat33d c) {
  c[0*3+0] = a[0*3+0] * b[0*3+0] + a[0*3+1] * b[1*3+0] + a[0*3+2] * b[2*3+0];
  c[0*3+1] = a[0*3+0] * b[0*3+1] + a[0*3+1] * b[1*3+1] + a[0*3+2] * b[2*3+1];
  c[0*3+2] = a[0*3+0] * b[0*3+2] + a[0*3+1] * b[1*3+2] + a[0*3+2] * b[2*3+2];

  c[1*3+0] = a[1*3+0] * b[0*3+0] + a[1*3+1] * b[1*3+0] + a[1*3+2] * b[2*3+0];
  c[1*3+1] = a[1*3+0] * b[0*3+1] + a[1*3+1] * b[1*3+1] + a[1*3+2] * b[2*3+1];
  c[1*3+2] = a[1*3+0] * b[0*3+2] + a[1*3+1] * b[1*3+2] + a[1*3+2] * b[2*3+2];

  c[2*3+0] = a[2*3+0] * b[0*3+0] + a[2*3+1] * b[1*3+0] + a[2*3+2] * b[2*3+0];
  c[2*3+1] = a[2*3+0] * b[0*3+1] + a[2*3+1] * b[1*3+1] + a[2*3+2] * b[2*3+1];
  c[2*3+2] = a[2*3+0] * b[0*3+2] + a[2*3+1] * b[1*3+2] + a[2*3+2] * b[2*3+2];
}

static inline
void mat33dmulvdot(mat33d _a, mat33d _b, mat33d _c) {
  double *a = (double*)_a;
  double *b = (double*)_b;
  double *c = (double*)_c;

  __m256d a0 = _mm256_maskload_pd(a, mask0111);
  __m256d a1 = _mm256_maskload_pd(a + 3, mask0111);
  __m256d a2 = _mm256_maskload_pd(a + 6, mask0111);
  __m256d b0 = _mm256_i64gather_pd(b, gatherMask036, 1);
  __m256d b1 = _mm256_i64gather_pd(b, gatherMask147, 1);
  __m256d b2 = _mm256_i64gather_pd(b, gatherMask258, 1);

  {
    __m256d temp0 = _mm256_mul_pd(a0, b0);
    __m256d temp1 = _mm256_mul_pd(a0, b1);
    __m256d temp2 = _mm256_mul_pd(a0, b2);
    __m256d temp3 = _mm256_hadd_pd(temp0, temp1);
    __m256d temp4 = _mm256_hadd_pd(temp2, temp2);
    __m256d temp5 = _mm256_permute2f128_pd(temp3, temp4, 0b00100000);
    __m256d temp6 = _mm256_permute2f128_pd(temp3, temp4, 0b00110001);
    __m256d res = _mm256_add_pd(temp5, temp6);
    _mm256_storeu_pd(c, res);
  } {
    __m256d temp0 = _mm256_mul_pd(a1, b0);
    __m256d temp1 = _mm256_mul_pd(a1, b1);
    __m256d temp2 = _mm256_mul_pd(a1, b2);
    __m256d temp3 = _mm256_hadd_pd(temp0, temp1);
    __m256d temp4 = _mm256_hadd_pd(temp2, temp2);
    __m256d temp5 = _mm256_permute2f128_pd(temp3, temp4, 0b00100000);
    __m256d temp6 = _mm256_permute2f128_pd(temp3, temp4, 0b00110001);
    __m256d res = _mm256_add_pd(temp5, temp6);
    _mm256_storeu_pd(c + 3, res);
  } {
    __m256d temp0 = _mm256_mul_pd(a2, b0);
    __m256d temp1 = _mm256_mul_pd(a2, b1);
    __m256d temp2 = _mm256_mul_pd(a2, b2);
    __m256d temp3 = _mm256_hadd_pd(temp0, temp1);
    __m256d temp4 = _mm256_hadd_pd(temp2, temp2);
    __m256d temp5 = _mm256_permute2f128_pd(temp3, temp4, 0b00100000);
    __m256d temp6 = _mm256_permute2f128_pd(temp3, temp4, 0b00110001);
    __m256d res = _mm256_add_pd(temp5, temp6);
    _mm256_maskstore_pd(c + 6, mask0111, res);
  }
}

static inline
void mat33dmulvshuf(mat33d _a, mat33d _b, mat33d _c) {
  double *a = (double*)_a;
  double *b = (double*)_b;
  double *c = (double*)_c;

  __m256d _0123 = _mm256_loadu_pd(a);
  __m256d _4567 = _mm256_loadu_pd(a + 4);
  __m256d _8888 = _mm256_broadcast_sd(a + 8);

  __m256d abcd = _mm256_loadu_pd(b);
  __m256d efgh = _mm256_loadu_pd(b + 4);
  __m256d iiii = _mm256_broadcast_sd(b + 8);

  __m256d abca = _mm256_permute4x64_pd(abcd, 0b00100100);
  __m256d _0003 = _mm256_permute4x64_pd(_0123, 0b11000000);
  __m256d c1 = _mm256_mul_pd(abca, _0003);

  __m256d efcd = _mm256_permute2f128_pd(abcd, efgh, 0b00010010);
  __m256d defd = _mm256_permute4x64_pd(efcd, 0b11010011);
  __m256d _1212 = _mm256_permute4x64_pd(_0123, 0b10011001);
  __m256d _1245 = _mm256_permute2f128_pd(_1212, _4567, 0b00100000);
  __m256d _1114 = _mm256_shuffle_pd(_1212, _1245, 0b0000);
  __m256d c2 = _mm256_mul_pd(defd, _1114);

  __m256d ifig = _mm256_shuffle_pd(iiii, efgh, 0b0010);
  __m256d ghig = _mm256_permute2f128_pd(ifig, efgh, 0b00010011);
  __m256d _2225 = _mm256_shuffle_pd(_1212, _1245, 0b1111);
  __m256d c3 = _mm256_mul_pd(_2225, ghig);

  _mm256_storeu_pd(c, _mm256_add_pd(_mm256_add_pd(c1, c2), c3));

  __m256d bcab = _mm256_permute4x64_pd(abca, 0b01001001);
  __m256d _2367 = _mm256_permute2f128_pd(_0123, _4567, 0b00110001);
  __m256d _3366 = _mm256_permute_pd(_2367, 0b0011);
  __m256d c4 = _mm256_mul_pd(bcab, _3366);

  __m256d efde = _mm256_permute4x64_pd(defd, 0b01001001);
  __m256d _4477 = _mm256_permute_pd(_4567, 0b1100);
  __m256d c5 = _mm256_mul_pd(efde, _4477);

  __m256d high = _mm256_permute4x64_pd(ghig, 0b01001001);
  __m256d _6755 = _mm256_permute4x64_pd(_4567, 0b01011110);
  __m256d _5588 = _mm256_permute2f128_pd(_6755, _8888, 0b00100001);
  __m256d c6 = _mm256_mul_pd(high, _5588);

  _mm256_storeu_pd(c + 4, _mm256_add_pd(_mm256_add_pd(c4, c5), c6));

  __m256d cfxx = _mm256_shuffle_pd(bcab, efcd, 0b0011);
  __m256d c7 = _mm256_mul_pd(cfxx, _6755);
  __m256d c8 = _mm256_permute_pd(c7, 0b1111);
  __m256d c9 = _mm256_mul_pd(iiii, _8888);

  _mm256_storeu_pd(c + 8, _mm256_add_pd(_mm256_add_pd(c7, c8), c9));
}

void init() {
  dummyfile = fopen("/dev/null", "w");
  mask0111 = _mm256_set_epi64x(0, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX);
  gatherMask036 = _mm256_set_epi64x(0, 48, 24, 0);
  gatherMask147 = _mm256_set_epi64x(0, 56, 32, 8);
  gatherMask258 = _mm256_set_epi64x(0, 64, 40, 16);
  srand(clock());
}

int main (int argc, char **argv) {
  if (argc > 1) {
    repeat = atoi(argv[1]);
  }
  init();

  if (1) {
    cout << "verify\n"
         << verify<vec4d, vec4d, vec4d>(vec3adds, vec3addsunroll, comp<3>, rand, rand)
        << verify<vec4d, vec4d, vec4d>(vec3adds, vec3addv, comp<3>, rand, rand)
        << verify<vec3d, vec3d, vec3d>(vec3adds, vec3addvmask, comp<3>, rand, rand)
        << verify<vec3d, vec3d, vec1d>(vec3dots, vec3dotsunroll, comp1, rand, rand)
        << verify<vec3d, vec3d, vec1d>(vec3dots, vec3dotv, comp1, rand, rand)
        << verify<vec3d, vec3d, vec1d>(vec3dots, vec3dotv2, comp1, rand, rand)
        << verify<vec3d, vec3d, vec1d>(vec3dots, vec3dotv3, comp1, rand, rand)
        << verify<vec3d, vec3d, mat33d>(vec3exts, vec3extsunroll, comp, rand, rand)
        << verify<vec3d, vec3d, mat33d>(vec3exts, vec3extv, comp, rand, rand)
        << verify<mat33d, vec3d, vec3d>(mat33vec3s, mat33vec3sunroll, comp<3>, rand, rand)
        << verify<mat33d, vec3d, vec3d>(mat33vec3s, mat33vec3vdot, comp<3>, rand, rand)
        << verify<mat33d, vec3d, vec3d>(mat33vec3s, mat33vec3vfused1, comp<3>, rand, rand)
        << verify<mat33d, vec3d, vec3d>(mat33vec3s, mat33vec3vfused2, comp<3>, rand, rand)
        << verify<mat33d, mat33d, mat33d>(mat33dmuls, mat33dmulsunroll, comp, rand, rand)
        << verify<mat33d, mat33d, mat33d>(mat33dmuls, mat33dmulvdot, comp, rand, rand)
        << verify<mat33d, mat33d, mat33d>(mat33dmuls, mat33dmulvshuf, comp<9>, rand, rand)
        ;
  }

  cout << "\nbenchmark\n";
  if (1) {
    benchmark(vec3d, vec3d, vec3d, vec3adds, rand, rand);
    benchmark(vec3d, vec3d, vec3d, vec3addsunroll, rand, rand);
    benchmark(vec4d, vec4d, vec4d, vec3addv, rand, rand);
    benchmark(vec3d, vec3d, vec3d, vec3addvmask, rand, rand);
    cout << endl;
  }
  if (1) {
    benchmark(vec3d, vec3d, vec1d, vec3dots, rand, rand);
    benchmark(vec3d, vec3d, vec1d, vec3dotsunroll, rand, rand);
    benchmark(vec3d, vec3d, vec1d, vec3dotv, rand, rand);
    benchmark(vec3d, vec3d, vec1d, vec3dotv2, rand, rand);
    benchmark(vec3d, vec3d, vec1d, vec3dotv3, rand, rand);
    cout << endl;
  }
  if (1) {
    benchmark(vec3d, vec3d, mat33d, vec3exts, rand, rand);
    benchmark(vec3d, vec3d, mat33d, vec3extsunroll, rand, rand);
    benchmark(vec3d, vec3d, mat33d, vec3extv, rand, rand);
    cout << endl;
  }
  if (1) {
    benchmark(mat33d, vec3d, vec3d, mat33vec3s, rand, rand);
    benchmark(mat33d, vec3d, vec3d, mat33vec3sunroll, rand, rand);
    benchmark(mat33d, vec3d, vec3d, mat33vec3vdot, rand, rand);
    benchmark(mat33d, vec3d, vec3d, mat33vec3vfused1, rand, rand);
    benchmark(mat33d, vec3d, vec3d, mat33vec3vfused2, rand, rand);
    cout << endl;
  }

  if (1) {
    benchmark(mat33d, mat33d, mat33d, mat33dmuls, rand, rand);
    benchmark(mat33d, mat33d, mat33d, mat33dmulsunroll, rand, rand);
    benchmark(mat33d, mat33d, mat33d, mat33dmulvdot, rand, rand);
    benchmark(mat33d, mat33d, mat33d, mat33dmulvshuf, rand, rand);
  }

  fclose(dummyfile);
  return 0;
}
