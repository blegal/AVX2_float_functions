
/*
 *	Shift and Rotate intrinsics for AVX2 - Copyright (c) 2017 Bertrand LE GAL
 *
 *  This software is provided 'as-is', without any express or
 *  implied warranty. In no event will the authors be held
 *  liable for any damages arising from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute
 *  it freely, subject to the following restrictions:
 *
 *  1. The origin of this software must not be misrepresented;
 *  you must not claim that you wrote the original software.
 *  If you use this software in a product, an acknowledgment
 *  in the product documentation would be appreciated but
 *  is not required.
 *
 *  2. Altered source versions must be plainly marked as such,
 *  and must not be misrepresented as being the original software.
 *
 *  3. This notice may not be removed or altered from any
 *  source distribution.
 *
 *  This source is based on the SHIFT functions presented:
 *  http://stackoverflow.com/questions/25248766/emulating-shifts-on-32-bytes-with-avx
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <cassert>
#include <complex>
#include <vector>

using namespace std;

#include <immintrin.h>

void MyAbsolute (const std::complex<float>* __restrict cplxIn,
                       float* __restrict absOut, const int length)
{
#ifdef __AVX2__
    const float* pIn1 = reinterpret_cast<const float*> (cplxIn + 0);
    const float* pIn2 = reinterpret_cast<const float*> (cplxIn + 4);
          float* pOut = absOut;

    for (int i = 0; i < length; i += 8)
    {
        const __m256 inLo = _mm256_loadu_ps ( pIn1 ); pIn1 += 16;
        const __m256 inHi = _mm256_loadu_ps ( pIn2 ); pIn2 += 16;

        const __m256 re  = _mm256_shuffle_ps (inLo, inHi, _MM_SHUFFLE (2, 0, 2, 0));
        const __m256 im  = _mm256_shuffle_ps (inLo, inHi, _MM_SHUFFLE (3, 1, 3, 1));
        const __m256 abs = _mm256_sqrt_ps (_mm256_add_ps (_mm256_mul_ps (re, re), _mm256_mul_ps (im, im)));

        const __m256d ordered = _mm256_permute4x64_pd (_mm256_castps_pd(abs), _MM_SHUFFLE(3,1,2,0));
        _mm256_storeu_ps (pOut, _mm256_castpd_ps(ordered));
        pOut += 8;
    }

    int start = (length >> 3) << 3;
    for (int i = start; i < length; i += 8)
    {
        float rr = cplxIn[i].real() * cplxIn[i].real() ;
        float ii = cplxIn[i].imag() * cplxIn[i].imag() ;
        float mod2 = rr + ii;
        float mod = sqrtf(mod2);
        absOut[i] = mod;
    }

    //
    //
    //
//    if( length & 0x07 )
//    {
//        printf("Oups il reste des données !\n");
//        exit( EXIT_FAILURE );
//    }
#endif
}


//
// On affiche les données floattantes scalaires ou complexes
//

inline void float_t_show(const vector<float>& liste)
{
	for(int i = 0; i < liste.size(); i++)
	{
		if( (i%8 == 0) && (i != 0) )
		{
			printf("\n");
		}
		printf("%1.3f ", liste.at(i));
	}
	printf("\n");
}

inline void float_t_show(const vector< complex<float> >& cplx)
{
	for(int i = 0; i < cplx.size(); i++)
	{
		if( (i%8 == 0) && (i != 0) )
			printf("\n");
		printf("(%6.3f, %6.3f) ", cplx[i].real(), cplx[i].imag());
	}
	printf("\n");
}


#define SIZE 32


int main(int argc, char* argv[])
{
	vector< complex<float> > A;
	
	for(int i = 0; i < SIZE; i += 1)
	{
		complex<float> c(i, i);
		A.push_back( c );
	}

	float_t_show( A );	
		

    return 1;
}
