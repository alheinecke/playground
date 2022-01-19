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

typedef unsigned short float16;
typedef union float_uint {
  float f;
  unsigned int u;
} float_uint;


float f16_to_f32( float16 in ) {
  unsigned int f32_bias = 127;
  unsigned int f16_bias = 15;
  unsigned int s = ( in & 0x8000 ) << 16;
  unsigned int e = ( in & 0x7c00 ) >> 10;
  unsigned int m = ( in & 0x03ff );
  unsigned int e_norm = e + (f32_bias - f16_bias);
  float_uint res;

  /* convert denormal fp16 number into a normal fp32 number */
  if ( (e == 0) && (m != 0) ) {
    unsigned int lz_cnt = 9;
    lz_cnt = ( m >   0x1 ) ? 8 : lz_cnt;
    lz_cnt = ( m >   0x3 ) ? 7 : lz_cnt;
    lz_cnt = ( m >   0x7 ) ? 6 : lz_cnt;
    lz_cnt = ( m >   0xf ) ? 5 : lz_cnt;
    lz_cnt = ( m >  0x1f ) ? 4 : lz_cnt;
    lz_cnt = ( m >  0x3f ) ? 3 : lz_cnt;
    lz_cnt = ( m >  0x7f ) ? 2 : lz_cnt;
    lz_cnt = ( m >  0xff ) ? 1 : lz_cnt;
    lz_cnt = ( m > 0x1ff ) ? 0 : lz_cnt;
    e_norm -= lz_cnt;
    m = (m << (lz_cnt+1)) & 0x03ff;
  } else if ( (e == 0) && (m == 0) ) {
    e_norm = 0;
  } else if ( e == 0x1f ) {
    e_norm = 0xff;
    m |= ( m == 0 ) ? 0 : 0x0200; /* making first mantissa bit 1 */
  }

  /* set result to 0 */
  res.u = 0x0;
  /* set exp and mant */
  res.u |= (e_norm << 23);
  res.u |= (m << 13);
  /* sign it */
  res.u |= s;

  return res.f;
}

float16 f32_to_f16( float in ) {
  unsigned int f32_bias = 127;
  unsigned int f16_bias = 15;
  float_uint hybrid_in;
  float16 res = 0;
  unsigned int s, e, m, e_f32, m_f32;
  unsigned int fixup;
  hybrid_in.f = in;

  /* DAZ */
  hybrid_in.u = ( (hybrid_in.u & 0x7f800000) == 0x0 ) ? ( hybrid_in.u & 0x80000000 ) : ( hybrid_in.u & 0xffffffff );

  s = ( hybrid_in.u & 0x80000000 ) >> 16;
  e_f32 = ( hybrid_in.u & 0x7f800000 ) >> 23;
  m_f32 = ( hybrid_in.u & 0x007fffff );

  /* RNE round */
  fixup = (m_f32 >> 13) & 0x1;
  hybrid_in.u = hybrid_in.u + 0x000000fff + fixup;
  e = ( hybrid_in.u & 0x7f800000 ) >> 23;
  m = ( hybrid_in.u & 0x007fffff );

  /* special value */
  if ( e_f32 == 0xff ) {
    e = 0x1f;
    m = (m_f32 == 0) ? 0 : (m_f32 >> 13) | 0x200;
  /* overflow */
  } else if ( e_f32 > (f32_bias + f16_bias) ) {
    e = 0x1f;
    m = 0x0;
  /* smaller than denormal f16 */
  } else if ( e_f32 < f32_bias - f16_bias - 10 ) {
    e = 0x0;
    m = 0x0;
  /* denormal */
  } else if ( e_f32 < f32_bias - f16_bias ) {
    m = m >> ( ( (f32_bias - f16_bias) - e ) + 13 );
    e = 0x0;
  /* normal */
  } else {
    e -= (f32_bias - f16_bias);
    m = m >> 13;
  }

  /* set result to 0 */
  res = 0x0;
  /* set exp and mant */
  res |= e << 10;
  res |= m;
  /* sign it */
  res |= s;

  return res;
}

void print_f16_f32( float16 f16, float f32 ) {
  float_uint hybrid; 
  hybrid.f= f32;
  unsigned int e_f16 = ( f16 & 0x7c00 ) >> 10;
  unsigned int m_f16 = ( f16 & 0x03ff );
  unsigned int e_f32 = ( hybrid.u & 0x7f800000 ) >> 23;
  unsigned int m_f32 = ( hybrid.u & 0x007fffff );

  printf( "fp: %e, fp16-hex: 0x%x, fp32-hex: 0x%x, fp16-exp: %u, fp16-mant: 0x%x, fp32-exp: %u, fp32-mant: 0x%x\n", f32, f16, hybrid.u, e_f16, m_f16, e_f32, m_f32 );   
}

void print_f16_f32_v2( float16 f16, float f32, int i ) {
  float_uint hybrid; 
  hybrid.f= f32;
  unsigned int e_f16 = ( f16 & 0x7c00 ) >> 10;
  unsigned int m_f16 = ( f16 & 0x03ff );
  unsigned int e_f32 = ( hybrid.u & 0x7f800000 ) >> 23;
  unsigned int m_f32 = ( hybrid.u & 0x007fffff );

  if ( e_f16 != 0 ) {  
    printf( "%i, fp: %e, fp16-hex: 0x%x, fp32-hex: 0x%x, fp16-exp: %u, fp16-mant: 0x%x, fp32-exp: %u, fp32-mant: 0x%x\n", i, f32, f16, hybrid.u, e_f16, m_f16, e_f32, m_f32 );   
  }
}

int main( int argc, char* argv[] ) {
  float x_f32;
  float16 x_f16;
  unsigned int i;

  srand48( clock() );

  /* testing random numbers */
  x_f32 = (float)drand48();

  printf("using _cvtss_sh for random f32 value\n");
  x_f16 = _cvtss_sh( x_f32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
  print_f16_f32( x_f16, x_f32 );

  printf("using f32_to_f16 for random f32 value\n");
  x_f16 = f32_to_f16( x_f32 );
  print_f16_f32( x_f16, x_f32 );
  printf("\n");

#if 0
  {
    float_uint hybrid;
    hybrid.u = 0xb87feff3;
    x_f16 = _cvtss_sh( hybrid.f,  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
    print_f16_f32_v2( x_f16, hybrid.f, 0 );
    x_f16 = f32_to_f16( hybrid.f );
    print_f16_f32_v2( x_f16, hybrid.f, 1 );
    printf("\n");
  }
  return 0;
#endif

  printf("testing all 2^32-1 combinations...\n");
  for ( i = 0; i < 0xffffffff; ++i ) {
    float_uint hybrid;
    float16 f16_a;
    float16 f16_b;

    hybrid.u = i;
    f16_a = _cvtss_sh( hybrid.f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC );
    f16_b = f32_to_f16( hybrid.f );
    if ( f16_a != f16_b ) {
      print_f16_f32_v2( f16_a, hybrid.f, 0 );
      print_f16_f32_v2( f16_b, hybrid.f, 1 );
#if 0
      break;
#endif
    }
  }

  /* test denormal f16 */
  printf("using _cvtsh_ss for 0x1 to 0x200 mantiss\n");
  x_f16 = 0x0001;
  for ( i = 0; i < 10; i++ ) { 
    x_f32 = _cvtsh_ss( x_f16 );
    print_f16_f32( x_f16, x_f32 );
    x_f16 = x_f16 << 1;
  }
  printf("\n");

  printf("using f16_to_f32 for 0x1 to 0x200 mantiss\n");
  x_f16 = 0x0001;
  for ( i = 0; i < 10; i++ ) { 
    x_f32 = f16_to_f32( x_f16 );
    print_f16_f32( x_f16, x_f32 );
    x_f16 = x_f16 << 1;
  }
  printf("\n");

  printf("using _cvtsh_ss for 0x1 to 0x3ff mantiss\n");
  x_f16 = 0x3ff;
  for ( i = 0; i < 10; i++ ) { 
    x_f32 = _cvtsh_ss( x_f16 );
    print_f16_f32( x_f16, x_f32 );
    x_f16 = x_f16 >> 1;
  }
  printf("\n");

  printf("using f16_to_f32 for 0x1 to 0x3ff mantiss\n");
  x_f16 = 0x3ff;
  for ( i = 0; i < 10; i++ ) { 
    x_f32 = f16_to_f32( x_f16 );
    print_f16_f32( x_f16, x_f32 );
    x_f16 = x_f16 >> 1;
  }
  printf("\n");

  /* test all f16 -> f32 values */
  printf("testing all 65535 combinations...\n");
  for ( i = 0; i < 0x10000; ++i ) {
    float_uint hybrid_b;
    float_uint hybrid_a;
    hybrid_a.f = _cvtsh_ss( (float16)i );;
    hybrid_b.f = f16_to_f32( (float16)i );;
    if ( hybrid_a.u != hybrid_b.u ) {
      printf("error for input 0x%x: 0x%x vs. 0x%x\n", i, hybrid_a.u, hybrid_b.u);
    }
  }
  printf("...done\n");

  return 0;
}
