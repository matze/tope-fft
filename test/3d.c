#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include <fftw3.h>
#include <time.h>

#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#ifndef TOPEFFT_H
#define TOPEFFT_H
#include "../src/topefft.h"
#endif

void plotInGnuplot(double *d, fftw_complex *o, cufftDoubleComplex *c, int n) {
	FILE *gplot = popen("gnuplot -persistent", "w");
	int i;
	#if 0
	#if 1
	fprintf(gplot, "set multiplot layout 1,3\n");
	fprintf(gplot, "plot '-' title 'topeFFT' w l lw 3 lc rgb 'red'\n");
	for (i = 0; i < n; i++)  
		fprintf(gplot, "%lf\n", pow(pow(d[2*i],2)+pow(d[2*i+1],2),0.5));
	fprintf(gplot, "e");
	#endif
	#if 0
	fprintf(gplot, "set origin 0,0\n");
	fprintf(gplot, "plot '-' title 'FFTW' w l lw 3 lc rgb 'blue'\n");
	for (i = 0; i < n; i++)  
		fprintf(gplot, "%lf\n", pow(pow(o[i][0],2)+pow(o[i][1],2),0.5));
	fprintf(gplot, "e");
	#endif
	#if 0
	fprintf(gplot, "set origin 0,0\n");
	fprintf(gplot, "plot '-' title 'cuFFT' w l lw 3 lc rgb 'green'\n");
	for (i = 0; i < n; i++) 
		fprintf(gplot, "%lf\n", pow(pow(c[i].x,2)+pow(c[i].y,2),0.5));
	fprintf(gplot, "e");
	#endif
	#endif
	#if 0
	FILE *temp = fopen("delete", "w");
	for (i = 0; i < n; i++) { 
		fprintf(temp, "%d\t%lf\t%lf\t%lf\n", 
						i, 
						pow(pow(d[2*i],2)+pow(d[2*i+1],2),0.5),
						pow(pow(o[i][0],2)+pow(o[i][1],2),0.5),
						pow(pow(c[i].x,2)+pow(c[i].y,2),0.5));
	}
	fprintf(gplot, 	"plot 'delete' using 1:2 title 'topeFFT' w l lw 10,"\
					"'delete' using 1:3 title 'FFTW' w l lw 6,"\
					"'delete' using 1:4 title 'cuFFT' w l lw 2\n");
	#endif
}


int main(int argc, char *argv[])
{
	int N = atoi(argv[1]);

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

	#if 0 // Show Output
	for (i = 0; i < N; i++) {
		printf("%lf:%lf\n", data[2*i], data[2*i+1]);
	}
	#endif

	#if 0 // Start Inverse
	tope1DExec(&framework, &plan, data, INVERSE);
	#if 1 // Show Output
	for (i = 0; i < N; i++) {
		printf("%lf:%lf\n", data[2*i], data[2*i+1]);
	}
	#endif
	#endif
	
	tope1DDestroy(&framework, &plan);
	#endif

	#if 0 /* FFTW Starts */
	fftw_complex *in, *out;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
	for (i = 0; i < N; i++) {
		in[i][0] = sin(2*PI*i/N);
		in[i][1] = 0;
	}

	struct timespec start,end;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	double t_ns = (double)(end.tv_sec - start.tv_sec) * 1.0e9 + (double)(end.tv_nsec - start.tv_nsec);

	fftw_destroy_plan(p);

	#if 0 // Show Output
	for (i = 0; i < N; i++) {
		printf("%lf:%lf\n", out[i][0], out[i][1]);
	}
	#endif
	#endif

	#if 0 /* Cuda FFT Starts */
	float cuTime;
	cudaEvent_t custart, custop;
	cufftHandle cudaPlan;
	cufftDoubleComplex *dataCuda;
	cufftDoubleComplex *dataLoca;
	dataLoca = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*N);
	cudaMalloc((void**)&dataCuda, sizeof(cufftDoubleComplex)*N*1);
	cufftPlan1d(&cudaPlan, N, CUFFT_Z2Z, 1);
	
	for (i = 0; i < N; i++) {
		dataLoca[i].x = sin(2*PI*i/N);
		dataLoca[i].y = 0;
	}
	cudaMemcpy(dataCuda, dataLoca, sizeof(cufftDoubleComplex)*N, cudaMemcpyHostToDevice);
	cudaEventCreate(&custart);
	cudaEventCreate(&custop);
	cudaEventRecord(custart, 0);
	cufftExecZ2Z(cudaPlan, dataCuda, dataCuda, CUFFT_FORWARD);
	cudaEventRecord(custop, 0);
	cudaEventSynchronize(custop);
	cudaEventElapsedTime(&cuTime, custart, custop);
	cudaMemcpy(dataLoca, dataCuda, sizeof(cufftDoubleComplex)*N, cudaMemcpyDeviceToHost);
	cufftDestroy(cudaPlan);
	cudaFree(dataCuda);
	#if 0 // Show Output
	for (i = 0; i < N; i++) {
		printf("%lf:%lf\n", dataLoca[i].x, dataLoca[i].y);
	}
	#endif
	#endif

	//plotInGnuplot(data, out, dataLoca, N);
	//printf("%d\t%f\t%f\t%f\t%d\n", N, ((double)1.0e-9)*(plan.totalKernel+plan.totalPreKernel), t_ns*1.0e-9, cuTime*10e-3, plan.radix);
	return 0; 
}

