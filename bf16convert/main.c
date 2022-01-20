/******************************************************************************
** Copyright (c) 2021-2022, Alexander Heinecke                               **
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
#include <time.h>
#include <x86intrin.h>

typedef unsigned short bfloat16;
typedef union float_uint {
  float f;
  unsigned int u;
} float_uint;

float _cvtsbh_ss( bfloat16 in ) {
  float res = 0.0f;
  __mmask16 mask = _mm512_int2mask( 0x1 );
  __m512i infmask = _mm512_set1_epi32( 0x7f800000 );
  __m512i signmask = _mm512_set1_epi32( 0x80000000 );
  __m256i vsrc = _mm256_maskz_loadu_epi16( mask, (void*)&in );
  __m512i vres = _mm512_slli_epi32( _mm512_cvtepi16_epi32( vsrc ), 16 );
  __m512i vsign = _mm512_and_epi32( signmask, vres );
  __mmask16 mask2 = _mm512_cmp_epi32_mask( _mm512_setzero_epi32(), _mm512_and_epi32( infmask, vres ), _MM_CMPINT_EQ );
  vres = _mm512_mask_mov_epi32( vres, mask2, _mm512_setzero_epi32() );
  vres = _mm512_or_epi32( vres, vsign );
  _mm512_mask_storeu_epi32( (void*)&res, mask, vres );
  return res;
}

bfloat16 _cvtness_sbh( float in ) {
  bfloat16 res = 0;
  __mmask16 mask = _mm512_int2mask( 0x1 );
  __m512 vsrc = _mm512_maskz_loadu_ps( mask, (void*)&in );
  __m256bh vres = _mm512_maskz_cvtneps_pbh( mask, vsrc );
  _mm256_mask_storeu_epi16( (void*)&res, mask, (__m256i)vres );
  return res;
}  

float bf16_to_f32( bfloat16 in ) {
  float_uint res;

  res.u = in;
  /* DAZ */
  res.u = ( (res.u & 0x7f80) == 0x0 ) ? res.u & 0x8000 : res.u; 
  res.u = res.u << 16;

  return res.f;
}

bfloat16 f32_to_bf16( float in ) {
  bfloat16 res = 0;
  float_uint hybrid_in;
  unsigned int fixup;

  hybrid_in.f = in;

  /* DAZ */
  hybrid_in.u = ( (hybrid_in.u & 0x7f800000) == 0x0 ) ? ( hybrid_in.u & 0x80000000 ) : hybrid_in.u;
  /* RNE round */
  fixup = (hybrid_in.u >> 16) & 1;
  /* we don't round inf and NaN */
  hybrid_in.u = ( (hybrid_in.u & 0x7f800000) == 0x7f800000 ) ? ( ((hybrid_in.u & 0x007fffff) == 0x0) ? hybrid_in.u : hybrid_in.u | 0x00400000 ) : hybrid_in.u + 0x00007fff + fixup;  
 
  res = hybrid_in.u >> 16;

  return res;
}

void print_bf16_f32( bfloat16 bf16, float f32, int i ) {
  float_uint hybrid; 
  hybrid.f= f32;
  unsigned int e_bf16 = ( bf16 & 0x7f80 ) >> 7;
  unsigned int m_bf16 = ( bf16 & 0x007f );
  unsigned int e_f32 = ( hybrid.u & 0x7f800000 ) >> 23;
  unsigned int m_f32 = ( hybrid.u & 0x007fffff );

  printf( "%i, fp: %e, bfp16-hex: 0x%x, fp32-hex: 0x%x, bfp16-exp: %u, bfp16-mant: 0x%x, fp32-exp: %u, fp32-mant: 0x%x\n", i, f32, bf16, hybrid.u, e_bf16, m_bf16, e_f32, m_f32 );   
}

int main( int argc, char* argv[] ) {
  float x_f32;
  bfloat16 x_bf16;
  unsigned int i;

  srand48( clock() );

  /* testing random numbers */
  x_f32 = (float)drand48();

  printf("\ntesting F32 -> BF16 for a random scalar [0:1]\n");
  printf("using _cvtss_sh for random f32 value\n");
  x_bf16 = _cvtness_sbh( x_f32 );
  print_bf16_f32( x_bf16, x_f32, 0 );

  printf("using f32_to_bf16 for random f32 value\n");
  x_bf16 = f32_to_bf16( x_f32 );
  print_bf16_f32( x_bf16, x_f32, 1 );
  printf("\n");

  printf("testing F32 -> BF16\n");
  printf("testing all 2^32-1 combinations...\n");
  for ( i = 0; i < 0xffffffff; ++i ) {
    float_uint hybrid;
    bfloat16 bf16_a;
    bfloat16 bf16_b;

    hybrid.u = i;
    bf16_a = _cvtness_sbh( hybrid.f );
    bf16_b = f32_to_bf16( hybrid.f );
    if ( bf16_a != bf16_b ) {
      print_bf16_f32( bf16_a, hybrid.f, 0 );
      print_bf16_f32( bf16_b, hybrid.f, 1 );
#if 0
      break;
#endif
    }
  }
  {
    float_uint hybrid;
    bfloat16 bf16_a;
    bfloat16 bf16_b;
    i = 0xffffffff;
    hybrid.u = i;
    bf16_a = _cvtness_sbh( hybrid.f );
    bf16_b = f32_to_bf16( hybrid.f );
    if ( bf16_a != bf16_b ) {
      print_bf16_f32( bf16_a, hybrid.f, 0 );
      print_bf16_f32( bf16_b, hybrid.f, 1 );
    }
  }
  printf("...done\n\n");

  /* test all f16 -> f32 values */
  printf("testing BF16 -> F32\n");
  printf("testing all 65535 combinations...\n");
  for ( i = 0; i < 0x10000; ++i ) {
    float_uint hybrid_b;
    float_uint hybrid_a;
    hybrid_a.f = _cvtsbh_ss( (bfloat16)i );;
    hybrid_b.f = bf16_to_f32( (bfloat16)i );;
    if ( hybrid_a.u != hybrid_b.u ) {
      print_bf16_f32( (bfloat16)i, hybrid_a.f, 0 );
      print_bf16_f32( (bfloat16)i, hybrid_b.f, 1 );
    }
  }
  printf("...done\n\n");

  return 0;
}
