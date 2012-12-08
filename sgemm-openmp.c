//Project 3-2
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <emmintrin.h>
#include <nmmintrin.h>
#include <omp.h>


void sgemm(int m, int n, float *A, float *C)
{


  int rem  = (m/32)*32;

  #pragma omp parallel for num_threads(16)
  for (int k = 0; k < m; k++) {
   
    float *Ckm = C+k*m;

    for (int i = 0; i < n; i++) {

      int im = i*m;

      float *Aim = A+im;

      __m128 s = _mm_load1_ps(A+im+k);

      for (int j = 0; j < rem; j+=32) {
        
              //Unroll #1
            __m128 c1 = _mm_loadu_ps(Ckm+j);
            __m128 a1 = _mm_loadu_ps(Aim+j);
            __m128 t1 = _mm_mul_ps(a1, s);
            t1 = _mm_add_ps(c1,t1);
            _mm_storeu_ps(Ckm+j,t1);

             //Unroll #2
            c1= _mm_loadu_ps(Ckm+j+4);
            a1= _mm_loadu_ps(Aim+j+4);
            t1 = _mm_mul_ps(a1, s);
            t1 = _mm_add_ps(c1,t1);
            _mm_storeu_ps(Ckm+j+4,t1);

             //Unroll #3
            c1=  _mm_loadu_ps(Ckm+j+8);
            a1= _mm_loadu_ps(Aim+j+8);
            t1 = _mm_mul_ps(a1, s);
            t1 = _mm_add_ps(c1,t1);
            _mm_storeu_ps(Ckm+j+8,t1);

             //Unroll #4
            c1= _mm_loadu_ps(Ckm+j+12);
            a1= _mm_loadu_ps(Aim+j+12);
            t1 = _mm_mul_ps(a1, s);
            t1 = _mm_add_ps(c1,t1);
            _mm_storeu_ps(Ckm+j+12,t1);

             //Unroll #5
            c1= _mm_loadu_ps(Ckm+j+16);
            a1= _mm_loadu_ps(Aim+j+16);
            t1 = _mm_mul_ps(a1, s);
            t1 = _mm_add_ps(c1,t1);
            _mm_storeu_ps(Ckm+j+16,t1);

             //Unroll #6
            c1= _mm_loadu_ps(Ckm+j+20);
            a1= _mm_loadu_ps(Aim+j+20);
            t1 = _mm_mul_ps(a1, s);
            t1 = _mm_add_ps(c1,t1);
            _mm_storeu_ps(Ckm+j+20,t1);

             //Unroll #7
            c1= _mm_loadu_ps(Ckm+j+24);
            a1= _mm_loadu_ps(Aim+j+24);
            t1 = _mm_mul_ps(a1, s);
            t1 = _mm_add_ps(c1,t1);
            _mm_storeu_ps(Ckm+j+24,t1);

             //Unroll #8
            c1= _mm_loadu_ps(Ckm+j+28);
            a1= _mm_loadu_ps(Aim+j+28);
            t1 = _mm_mul_ps(a1, s);
            t1 = _mm_add_ps(c1,t1);
            _mm_storeu_ps(Ckm+j+28,t1);
            
          
      }
      for (int j = rem; j < m; j++) {
              C[k*m+j] += A[i*m+j] * A[k+i*m]; //Edge Case
      }
    }
  }
}
