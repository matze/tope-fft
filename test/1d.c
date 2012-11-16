#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include <fftw3.h>

#ifndef TOPEFFT_H
#define TOPEFFT_H
#include "../src/topefft.h"
#endif

int main()
{
	int N = 128;
	
	double *data = calloc(N*2,sizeof(double));

	double PI = acos(-1);
	int i;
	for (i = 0; i < N; i++) {
		data[2*i] = i+1;//sin(2*PI*i/N);
	}

	#if 1 /* Tope FFT Starts */
	struct topeFFT framework;
	topeFFTInit(&framework);

	struct topePlan1D plan;
	tope1DPlanInit(&framework, &plan, N, C2C, data);
	tope1DExec(&framework, &plan, data, FORWARD);

	#if 1
	for (i = 0; i < N; i++) {
		printf("%lf:%lf\n", data[2*i], data[2*i+1]);
	}
	#endif

	#endif

	#if 1 /* FFTW Starts */
	fftw_complex *in, *out;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
	for (i = 0; i < N; i++) {
		in[i][0] = i+1;//sin(2*PI*i/N);
		in[i][1] = 0;
	}
	fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);

	#if 0
	for (i = 0; i < N; i++) {
		printf("%lf:%lf\n", out[i][0], out[i][1]);
	}
	#endif

	#endif

	plotInGnuplot(data, out, N);
}

