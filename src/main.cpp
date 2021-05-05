
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

using namespace std;
using namespace avx2_func;

inline void float_t_show(const vector<float>& table)
{
	for(int i = 0; i < 32; i++)
	{
		if( (i%8 == 0) && (i != 0) )
			printf("\n");
		printf("%1.3f ", table[i]);
	}
	printf("\n");
}

inline void float_t_show(const vector< complex<float> >& cplx)
{
	for(int i = 0; i < 32; i++)
	{
		if( (i%4 == 0) && (i != 0) )
			printf("\n");
		printf("(%1.3f, %1.3f) ", table[i].real(), table[i].imag());
	}
	printf("\n");
}



int main(int argc, char* argv[]) {

    return 1;
}
