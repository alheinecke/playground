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

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

int main( int argc, char* argv[] ) {
  char as[2][16][4];       /* A is in VNNI format, it's a [k][m] = [ 8][16] matrix */
  char bs[2][8];           /* B is in norm foramt, it's a [n][k] = [ 2][ 8] matrix */
  int a_fixup[16];         /* A FixUp is a vector reduce along k */
  unsigned char bu[2][8];  /* B is in norm foramt, it's a [n][k] = [ 2][ 8] matrix */
  int css[2][16];          /* C is in norm format, is's a [n][m] = [ 2][16] matrix */
  int csu[2][16];
  int m, n, k1, k2, k;

  srand(time(NULL));

  /* init A */
  for ( k1 = 0; k1 < 2; ++k1 ) {
    for ( m = 0; m < 16; ++m ) {
      for ( k2 = 0; k2 < 4; ++k2 ) {
        as[k1][m][k2] = (k2 % 2 == 0) ? (char)(rand() % 100) : (char)((rand() % 100) * -1);
      }
    }
  }
  /* compute a_fixup */
  for ( m = 0; m < 16; ++m ) {
    a_fixup[m] = 0;
    for ( k1 = 0; k1 < 2; ++k1 ) {
      for ( k2 = 0; k2 < 4; ++k2 ) {
        a_fixup[m] += as[k1][m][k2] * 128;
      }
    }
  }

  /* init B */
  for ( n = 0; n < 2; ++n ) {
    for ( k = 0; k < 8; ++k ) {
      bs[n][k] = (k % 2 == 0) ? (char)(rand() % 100) : (char)((rand() % 100) * -1);
      bu[n][k] = (unsigned char)(bs[n][k] + 128);
    }
  }

  /* matmul */
  for ( n = 0; n < 2; ++n ) {
    for ( m = 0; m < 16; ++m ) {
      css[n][m] = 0;
      csu[n][m] = 0;
      for (k1 = 0; k1 < 2; ++k1) {
        for (k2 = 0; k2 < 4; ++k2) {
          css[n][m] += (int)as[k1][m][k2] * (int)bs[n][(k1 * 4) + k2];
          csu[n][m] += (int)as[k1][m][k2] * (int)bu[n][(k1 * 4) + k2];
        }
      }
    }
  }

  /* print results */
  for (m = 0; m < 16; ++m) {
    for (n = 0; n < 2; ++n)  printf("%i ", css[n][m]);
    printf("     ");
    for (n = 0; n < 2; ++n)  printf("%i ", csu[n][m]);
    printf("     ");
    for (n = 0; n < 2; ++n)  printf("%i ", csu[n][m] - a_fixup[m]);
    printf("     ");
    for (n = 0; n < 2; ++n)  printf("%i ", css[n][m] - (csu[n][m] - a_fixup[m]));
    printf("\n");
  }

  return 0;
}
