#include <time.h>
#include <stdlib.h>
#include <stdio.h>

int main( int argc, char* argv[] ) {
  char as[2][16][4];       /* A is in VNNI format, it's a [k][m] = [ 8][16] matrix */
  char bs[2][8];           /* B is in norm foramt, it's a [n][k] = [ 2][ 8] matrix */
  int a_fixup[16]          /* A FixUp is a vector reduce along k */
  unsigned char bu[2][8];  /* B is in norm foramt, it's a [n][k] = [ 2][ 8] matrix */
  int css[16];             /* C is in norm format, is's a [n][m] = [ 2][16] matrix */
  int csu[16];
  int m, n, k1, k2, k;

  srand(time(NULL));

  /* init A */
  for (k1 = 0; k1 < 2; ++k1) {
    for (m = 0; m < 16; ++m) {
      for (k2 = 0; k2 < 4; ++k2) {
        a[k1][m][k2] = (k2 % 2 == 0) ? (char)(rand() % 100) : (char)((rand() % 100) * -1);
      }
    }
  }
  /* compute a_fixup */
  for (m = 0; m < 16; ++m) {
    a_fixup[m] = 0;
    for (k1 = 0; k1 < 2; ++k1) {
      for (k2 = 0; k2 < 4; ++k2) {
        a_fixup[m] += a[k1][m][k2] * 128;
      }
    }
  }

  /* init B */
  for (n = 0; n < 2; ++n) {
      for (k = 0; k < 8; ++k) {
          bs[n][k] = (k % 2 == 0) ? (char)(rand() % 100) : (char)((rand() % 100) * -1);
          bu[n][k] = (unsigned char)(bs[i] + 128);
      }
  }

  /* matmul */
  for (n = 0; n < 2; ++n) {
      for (m = 0; m < 16; ++m) {
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
      print("     ");
      for (n = 0; n < 2; ++n)  printf("%i ", csu[n][m]);
      print("     ");
      for (n = 0; n < 2; ++n)  printf("%i ", csu[n][m] - a_fixup[m]);
      print("     ");
      for (n = 0; n < 2; ++n)  printf("%i ", css[n][m] - (csu[n][m] - a_fixup[m]));
      print("\n");
  }

  return 0;
}
