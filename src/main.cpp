
/*
 *	Shift and Rotate intrinsics for AVX2 - Copyright (c) 2017 Bertrand LE
 *GAL
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
#include <cassert>
#include <complex>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

#include <immintrin.h>

inline void float_t_show(const __m256 reg) {
    float values[8];
    _mm256_storeu_ps(values, reg);
    for (int i = 0; i < 8; i++) {
        printf("%7.3f ", values[i]);
    }
    printf("\n");
}

void cplxModule(const std::complex<float> * __restrict cplxIn,
                float * __restrict absOut, const int length) {
    const float * pIn1 = reinterpret_cast<const float *>(cplxIn + 0);
    const float * pIn2 = reinterpret_cast<const float *>(cplxIn + 4);
    float *       pOut = absOut;

    for (int i = 0; i < length; i += 8) {
        const __m256 inLo = _mm256_loadu_ps(pIn1);
        pIn1 += 16;
        const __m256 inHi = _mm256_loadu_ps(pIn2);
        pIn2 += 16;

        const __m256 re  = _mm256_shuffle_ps(inLo, inHi, _MM_SHUFFLE(2, 0, 2, 0));
        const __m256 im  = _mm256_shuffle_ps(inLo, inHi, _MM_SHUFFLE(3, 1, 3, 1));
        const __m256 abs = _mm256_sqrt_ps(
            _mm256_add_ps(_mm256_mul_ps(re, re), _mm256_mul_ps(im, im)));

        const __m256d ordered =
            _mm256_permute4x64_pd(_mm256_castps_pd(abs), _MM_SHUFFLE(3, 1, 2, 0));
        _mm256_storeu_ps(pOut, _mm256_castpd_ps(ordered));
        pOut += 8;
    }
}

void cplxMult(const std::complex<float> * __restrict cplxIn,
              const std::complex<float> * __restrict cplxMul,
              std::complex<float> * __restrict cplxOu, const int length) {
    const float * pIn  = reinterpret_cast<const float *>(cplxIn);
    const float * pMul = reinterpret_cast<const float *>(cplxMul);
    float *       pOut = reinterpret_cast<float *>(cplxOu);

    const __m256 b     = _mm256_loadu_ps(pMul);
    const __m256 bSwap = _mm256_shuffle_ps(b, b, 0xB1);

    for (int i = 0; i < length; i += 8) {
        const __m256 a         = _mm256_loadu_ps(pIn + i);
        const __m256 aRe       = _mm256_shuffle_ps(a, a, 0xA0);
        const __m256 aIm       = _mm256_shuffle_ps(a, a, 0xF5);
        const __m256 aIm_bSwap = _mm256_mul_ps(aIm, bSwap);
        const __m256 aRe_b     = _mm256_mul_ps(aRe, b);
        const __m256 res       = _mm256_addsub_ps(aRe_b, aIm_bSwap);

        printf("b         : ");
        float_t_show(b);
        printf("bSwap     : ");
        float_t_show(bSwap);
        printf("a         : ");
        float_t_show(a);
        printf("aRe       : ");
        float_t_show(aRe);
        printf("aIm       : ");
        float_t_show(aIm);
        printf("aIm_bSwap : ");
        float_t_show(aIm_bSwap);
        printf("aRe_b     : ");
        float_t_show(aRe_b);
        printf("res       : ");
        float_t_show(res);
        printf("\n");

        _mm256_storeu_ps(pOut + i, res);
    }
}

//
// On affiche les données floattantes scalaires ou complexes
//

inline void float_t_show(const vector<float> & liste) {
    for (int i = 0; i < liste.size(); i++) {
        if ((i % 8 == 0) && (i != 0)) {
            printf("\n");
        }
        printf("%7.3f ", liste.at(i));
    }
    printf("\n");
}

inline void float_t_show(const vector<complex<float>> & cplx) {
    for (int i = 0; i < cplx.size(); i++) {
        if ((i % 8 == 0) && (i != 0))
            printf("\n");
        printf("(%7.3f, %7.3f) ", cplx[i].real(), cplx[i].imag());
    }
    printf("\n");
}

#define SIZE 32

// vec_alpha (F x complex -> F x RE IM)
float vec_alpha[] = {0.091892, 2.505220, 1.487593, 1.887682, 2.156101,
                     0.499669, 1.773584, -1.009793, 0.510125, -1.935473,
                     -1.050093, -1.847128, -2.183088, -0.791543, -2.365042,
                     0.729482, -1.518261, 1.99485};

// pn (q chips)
float vec_pn[64] = {1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1,
                    1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1,
                    -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1,
                    1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1,
                    1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1};

float accu_iq[] = {
    86.609744, -324.281624, -82.301513, 41.211150, 51.021555,
    57.169758, -82.649047, -168.527773, 480.233588, -279.785593,
    377.974147, -139.589307, 126.887525, -93.715722, 87.607555,
    -29.674300, 94.127395, 18.441327, 271.248266, 113.136763,
    -147.493468, -33.401906, 166.348989, 47.039689, 144.116217,
    -43.008931, 303.823655, -7.738545, -43.831192, -64.520394,
    -2.314234, 50.772556, -222.934220, -53.055202, 377.646156,
    50.584161, -338.853016, -79.423402, 12.840392, 300.247879,
    230.052943, -53.194555, 189.858111, 170.121864, -217.220112,
    146.431472, -68.519507, 215.254238, -331.212823, 55.752144,
    57.216807, 190.383446, 32.592598, -307.946254, -247.405869,
    -135.624135, -52.847041, 182.466004, 329.236616, -127.898405,
    59.520714, -295.671480, 152.164468, 20.144230, -169.437128,
    286.739289, 104.238827, 372.059523, 189.006157, 3.086682,
    -136.423554, 84.952120, -541.703678, 285.428722, 457.694315,
    787.385806, -62.593073, -86.639712, -414.275175, -164.735481,
    -629.516791, -494.414093, -75.974184, -58.044645, 92.055180,
    -26.410714, -164.962583, 26.903605, -459.325259, 328.875004,
    -438.698432, -113.000502, -131.905148, -152.486958, -292.931216,
    94.804412, 218.910235, -57.710518, 95.893032, -557.300484,
    -21.252229, 181.320489, 161.655410, 189.191674, 144.898745,
    574.181955, -573.484201, 16.427065, 353.049751, 633.569625,
    -60.816427, 101.933261, 461.974803, -107.154033, -136.356541,
    -737.208810, 207.267106, -272.806077, 330.566933, -24.889156,
    255.320290, -39.129335, -626.220646, -293.783429, -108.626377,
    65.774371, 178.426458, -269.018809};

float accu_real[64];
float accu_imag[64];

template <int N>
float process(float * __restrict cplx, float * __restrict tab_pn,
              float * __restrict acc_real, float * __restrict acc_imag) {
                  static_assert(N % 2 == 0, "N must be a power of 2!");
    __asm volatile("# LLVM-MCA-BEGIN bar");
    const __m256 cReal = _mm256_broadcast_ss(cplx);     // valeur de alpha real
    const __m256 cImag = _mm256_broadcast_ss(cplx + 1); // valeur de alpha imag
    __m256       mMax  = _mm256_setzero_ps();

#pragma unroll
    for (int i = 0; i < N; i += (sizeof(__m256) / sizeof(float))) {
        const __m256 rPN = _mm256_loadu_ps(
            tab_pn + i); // On charge les +1/-1 contenus dans le tableau PN

        const __m256 tReal =
            _mm256_mul_ps(cReal, rPN); // On multiplie les parties reelles
        const __m256 tImag =
            _mm256_mul_ps(cImag, rPN); // On multiplie les parties imags

        const __m256 aReal = _mm256_loadu_ps(
            acc_real + i); // chargement des données contenues dans l'accu
        const __m256 aImag = _mm256_loadu_ps(
            acc_imag + i); // chargement des données contenues dans l'accu

        const __m256 rReal =
            _mm256_add_ps(tReal, aReal); // On additionne les reals avec les reals
        const __m256 rImag =
            _mm256_add_ps(tImag, aImag); // On additionne les imags avec les imags

        _mm256_storeu_ps(acc_real + i, rReal); // memorisation du resultat de l'accu
        _mm256_storeu_ps(acc_imag + i, rImag); // memorisation du resultat de l'accu

        // On calcule le module du complexe

        const __m256 a2 = _mm256_mul_ps(rReal, rReal); //      A^2
        const __m256 b2 = _mm256_mul_ps(rImag, rImag); //            B^2
        const __m256 c2 = _mm256_add_ps(a2, b2);       //      A^2 - B^2
        const __m256 c  = _mm256_sqrt_ps(c2);          // SQRT(A^2 - B^2)

        mMax = _mm256_max_ps(mMax, c);
    }

    // On calcule le maximum des 8 floats

    const __m256 permHalves = _mm256_permute2f128_ps(mMax, mMax, 1);
    const __m256 m0         = _mm256_max_ps(permHalves, mMax);
    const __m256 perm0      = _mm256_permute_ps(m0, 0b01001110);
    const __m256 m1         = _mm256_max_ps(m0, perm0);
    const __m256 perm1      = _mm256_permute_ps(m1, 0b10110001);
    const __m256 m2         = _mm256_max_ps(perm1, m1);
    return _mm256_cvtss_f32(m2);
    __asm volatile("# LLVM-MCA-END bar");
}


int main(int argc, char * argv[]) {

    constexpr unsigned N = 64;

    for (int i = 0; i < N; i += 1) {
        accu_real[i] =
            accu_iq[2 * i + 0];            // Il me faut un tableau avec les reels et
        accu_imag[i] = accu_iq[2 * i + 1]; // un tableau avec imaginaires
    }

    float maxv = process<N>(vec_alpha, vec_pn, accu_real, accu_imag);

    cout << "valeur du max = " << maxv << endl;

#if 0
	vector< complex<float> > A;
	
	for(int i = 0; i < SIZE; i += 1)
	{
		complex<float> c(i, -i);
		A.push_back( c );
	}

    std::cout << std::endl << "Données initiales : " << std::endl;
	float_t_show( A );	
	
	vector< float > B( A.size() );

    cplxModule(A.data(), B.data(), A.size() );

    std::cout << std::endl << "Calcul du module : " << std::endl;
	float_t_show( B );

    std::complex<float> otherCplx[4];
    const __m256 c = _mm256_set_ps(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    _mm256_storeu_ps ((float*)otherCplx, c);

	vector< complex<float> > C;
    C.resize( A.size() );

    cplxMult(A.data(), otherCplx, C.data(), A.size() );

    std::cout << std::endl << "Resultat multiplication : " << std::endl;
	float_t_show( C );
#endif
    return EXIT_SUCCESS;
}
