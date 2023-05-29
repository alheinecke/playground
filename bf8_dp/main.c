/******************************************************************************
** Copyright (c) 2022, Alexander Heinecke                                    **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <libxsmm.h>

/*
 * Build Instructions:
 * gcc -I./../../LIBXSMM_workspace/libxsmm_github/include/ -L./../../LIBXSMM_workspace/libxsmm_github/lib/ -o main main.c -lxsmm -lxsmmnoblas -lm -ldl
 */

void run_f32( const float* a, const float* b, float* c ) {
  unsigned int i, j;

  for ( i = 0; i < 16; ++i ) {
    for ( j = 0; j < 4; ++j ) {
      c[i] += a[(i*4)+j] * b[j];
    }
  }
}

void run_sw( const libxsmm_bfloat8* a, const libxsmm_bfloat8* b, float* c ) {
  unsigned int i, j;
  float a_f32;
  float b_f32;

  for ( i = 0; i < 16; ++i ) {
    for ( j = 0; j < 4; ++j ) {
      libxsmm_convert_bf8_f32( &(a[(i*4)+j]), &a_f32, 1);
      libxsmm_convert_bf8_f32( &(b[j]), &b_f32, 1);
      
      c[i] += a_f32 * b_f32;
    }
  }
}


void run_asm( libxsmm_bfloat8* a, libxsmm_bfloat8* b, float* c ) {
  unsigned short perm[32] = { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 };
  unsigned short* perm_ptr = &(perm[0]);
  /* running replacement code on AVX512 stack */
  __asm__ __volatile__(
      "movq %0, %%rsi\n\t"                              /* addr a vector (32 elements) */
      "movq %1, %%rdx\n\t"                              /* addr b tuple (2 elements) */
      "movq %2, %%r12\n\t"                              /* addr c vector (16 elements)  */
      "movq %3, %%r14\n\t"
      "movq $0x2222222222222222, %%r13\n\t"             /* load mask for 16bit */
      "kmovq %%r13, %%k1\n\t"
      "movq $0x4444444444444444, %%r13\n\t"             /* load mask for 16bit */
      "kmovq %%r13, %%k2\n\t"
      "movq $0x8888888888888888, %%r13\n\t"             /* load mask for 16bit */
      "kmovq %%r13, %%k3\n\t"
      "vmovdqu16 0(%%r14), %%zmm5\n\t"                  /* load permute mask */
      "vmovups 0(%%r12), %%zmm4\n\t"                    /* load c */
      "vmovdqu8 0(%%rsi), %%zmm0\n\t"                   /* load a into zmm0 */
      "vpslld $8, %%zmm0, %%zmm1\n\t"                   /* move lower 8 bits into bits 15:8 to create FP16 fro BF8 */
      "vpermw %%zmm1, %%zmm5, %%zmm1\n\t"               /* pack  values together */
      "vcvtph2ps %%ymm1, %%zmm1\n\t"                    /* convert FP16 to FP32 */
      "vpbroadcastb 0(%%rdx), %%ymm2\n\t"               /* broadcast bf8 b[0] value values into zmm2 */ 
      "vpsllw $8, %%ymm2, %%ymm2\n\n"                   /* move lower 8 bits into bits 15:8 to create FP16 fro BF8 */
      "vcvtph2ps %%ymm2, %%zmm2\n\t"                    /* convert FP16 to FP32 */
      "vfmadd231ps %%zmm1, %%zmm2, %%zmm4\n\t"          /* multiply "0" elements of a and b and add to zmm4 */
      "vmovdqu8 %%zmm0, %%zmm1%{%%k1%}%{z%}\n\t"
      "vpermw %%zmm1, %%zmm5, %%zmm1\n\t"               /* pack  values together */
      "vcvtph2ps %%ymm1, %%zmm1\n\t"                    /* convert FP16 to FP32 */
      "vpbroadcastb 1(%%rdx), %%ymm2\n\t"               /* broadcast bf8 b[1] value values into zmm2 */ 
      "vpsllw $8, %%ymm2, %%ymm2\n\n"                   /* move lower 8 bits into bits 15:8 to create FP16 fro BF8 */
      "vcvtph2ps %%ymm2, %%zmm2\n\t"                    /* convert FP16 to FP32 */
      "vfmadd231ps %%zmm1, %%zmm2, %%zmm4\n\t"          /* multiply "1" elements of a and b and add to zmm4 */
      "vmovdqu8 %%zmm0, %%zmm1%{%%k2%}%{z%}\n\t"
      "vpsrld $8, %%zmm1, %%zmm1\n\t"                   /* move lower 8 bits into bits 15:8 to create FP16 fro BF8 */
      "vpermw %%zmm1, %%zmm5, %%zmm1\n\t"               /* pack  values together */
      "vcvtph2ps %%ymm1, %%zmm1\n\t"                    /* convert FP16 to FP32 */
      "vpbroadcastb 2(%%rdx), %%ymm2\n\t"               /* broadcast bf8 b[2] value values into zmm2 */ 
      "vpsllw $8, %%ymm2, %%ymm2\n\n"                   /* move lower 8 bits into bits 15:8 to create FP16 fro BF8 */
      "vcvtph2ps %%ymm2, %%zmm2\n\t"                    /* convert FP16 to FP32 */
      "vfmadd231ps %%zmm1, %%zmm2, %%zmm4\n\t"          /* multiply "2" elements of a and b and add to zmm4 */
      "vmovdqu8 %%zmm0, %%zmm1%{%%k3%}%{z%}\n\t"
      "vpsrld $16, %%zmm1, %%zmm1\n\t"                  /* move lower 8 bits into bits 15:8 to create FP16 fro BF8 */
      "vpermw %%zmm1, %%zmm5, %%zmm1\n\t"               /* pack  values together */
      "vcvtph2ps %%ymm1, %%zmm1\n\t"                    /* convert FP16 to FP32 */
      "vpbroadcastb 3(%%rdx), %%ymm2\n\t"               /* broadcast bf8 b[3] value values into zmm2 */ 
      "vpsllw $8, %%ymm2, %%ymm2\n\n"                   /* move lower 8 bits into bits 15:8 to create FP16 fro BF8 */
      "vcvtph2ps %%ymm2, %%zmm2\n\t"                    /* convert FP16 to FP32 */
      "vfmadd231ps %%zmm1, %%zmm2, %%zmm4\n\t"          /* multiply "3" elements of a and b and add to zmm4 */
      "vmovups %%zmm4, 0(%%r12)\n\t"                    /* store c vector */
  :: "m" (a), "m" (b), "m" (c), "m" (perm_ptr) 
  : "memory", "rsi", "rdx", "r12", "r13", "r14", "k1", "k2", "k3", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5" );
}


int main( int argc, char* argv[] ) {
  float c0[16];
  float c1[16];
  float c2[16];
  float c3[16];
  libxsmm_bfloat8 a[64];
  libxsmm_bfloat8 b[4];
  float f_a[64];
  float f_b[4];
  float bf8_a[64];
  float bf8_b[4];
  unsigned int i;

  for ( i = 0; i < 64; ++i ) {
    float data;
    data = (float)drand48();
    f_a[i] = data;
    libxsmm_rne_convert_fp32_bf8( &data, &(a[i]), 1);
    libxsmm_convert_bf8_f32( &(a[i]), &(bf8_a[i]), 1);
  }

  for ( i = 0; i < 4; ++i ) {
    float data;
    data = (float)drand48();
    f_b[i] = data;
    libxsmm_rne_convert_fp32_bf8( &data, &(b[i]), 1);
    libxsmm_convert_bf8_f32( &(b[i]), &(bf8_b[i]), 1);
  }

  for ( i = 0; i < 16; ++i ) {
    c0[i] = (float)drand48();
    c1[i] = c0[i];
    c2[i] = c0[i];
    c3[i] = c0[i];
  }
                    
  run_f32( f_a, f_b, c2 );
  run_f32( bf8_a, bf8_b, c3 );
  run_sw( a, b, c0 );
  run_asm( a, b, c1 );

  for (i = 0; i < 16; ++i ) { 
    printf("%f ", c2[i] );
  }
  printf("\n");
  for (i = 0; i < 16; ++i ) { 
    printf("%f ", c3[i] );
  }
  printf("\n");
  for (i = 0; i < 16; ++i ) { 
    printf("%f ", c0[i] );
  }
  printf("\n");
  for (i = 0; i < 16; ++i ) { 
    printf("%f ", c1[i] );
  }
  printf("\n");

  return 0;
}

