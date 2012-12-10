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
	#if 1 // Plot from memory
	#if 1
	fprintf(gplot, "set multiplot layout 1,3\n");
	fprintf(gplot, "plot '-' title 'topeFFT' w l lw 3 lc rgb 'red'\n");
	for (i = 0; i < n; i++)  
		fprintf(gplot, "%lf\n", pow(pow(d[2*i],2)+pow(d[2*i+1],2),0.5));
	fprintf(gplot, "e");
	#endif
	#if 1
	fprintf(gplot, "set origin 0,0\n");
	fprintf(gplot, "plot '-' title 'FFTW' w l lw 3 lc rgb 'blue'\n");
	for (i = 0; i < n; i++)  
		fprintf(gplot, "%lf\n", pow(pow(o[i][0],2)+pow(o[i][1],2),0.5));
	fprintf(gplot, "e");
	#endif
	#if 1
	fprintf(gplot, "set origin 0,0\n");
	fprintf(gplot, "plot '-' title 'cuFFT' w l lw 3 lc rgb 'green'\n");
	for (i = 0; i < n; i++) 
		fprintf(gplot, "%lf\n", pow(pow(c[i].x,2)+pow(c[i].y,2),0.5));
	fprintf(gplot, "e");
	#endif
	#endif
	#if 0 // Plot from file
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
	return;
}


int main(int argc, char *argv[])
{
	if (argc!=4) { printf("<exec> NX NY NZ\n"); exit(1); }
	int NX = atoi(argv[1]);
	int NY = atoi(argv[2]);
	int NZ = atoi(argv[3]);

	int i,j,k;
	int count = 0;
	double PI = acos(-1);
	
	#if 1 /* Tope FFT Starts */
	double *data = calloc(NX*NY*NZ*2,sizeof(double));

	for (k = 0; k < NZ; k++) {
		for (j = 0; j < NY; j++) {
			for (i = 0; i < NX; i++) {
				data[2*(k*NX*NY+j*NX+i)] = sin(2*PI*count/(NX*NY*NZ));
				count++;
			}
		}
	}

	struct topeFFT framework;
	topeFFTInit(&framework);

	struct topePlan3D plan;
	tope3DPlanInit(&framework, &plan, NX, NY, NZ, C2C, data);
	tope3DExec(&framework, &plan, data, FORWARD);

	#if 0 // Show Output
	for (k = 0; k < NZ; k++) {
		for (i = 0; i < NX; i++) {
			for (j = 0; j < NY; j++) {
				printf("%.4f %.4f\t", data[2*(k*NX*NY+j*NX+i)], data[2*(k*NX*NY+j*NX+i)+1]);
			}
			printf("\n");
		}
		printf("---\n");
	}
	#endif

	#if 0 // Start Inverse
	tope3DExec(&framework, &plan, data, INVERSE);
	#if 1 // Show Output
	for (k = 0; k < NZ; k++) {
		for (i = 0; i < NX; i++) {
			for (j = 0; j < NY; j++) {
				printf("%.4f %.4f\t", data[2*(k*NX*NY+j*NX+i)], data[2*(k*NX*NY+j*NX+i)+1]);
			}
			printf("\n");
		}
		printf("---\n");
	}
	#endif
	#endif
	
	tope3DDestroy(&framework, &plan);
	#endif

	#if 1 /* FFTW Starts */
	fftw_complex *in, *out;
	count = 0;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*NX*NY*NZ);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*NX*NY*NZ);
	for (k = 0; k < NZ; k++) {
		for (j = 0; j < NY; j++) {
			for (i = 0; i < NX; i++) {
				in[k*NX*NY+j*NX+i][0] = sin(2*PI*count/(NX*NY*NZ));
				in[k*NX*NY+j*NX+i][1] = 0;
				count++;
			}
		}
	}

	struct timespec start,end;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	fftw_plan p = fftw_plan_dft_3d(NZ, NY, NX, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	double t_ns = (double)(end.tv_sec - start.tv_sec) * 1.0e9 + (double)(end.tv_nsec - start.tv_nsec);

	fftw_destroy_plan(p);

	#if 0 // Show Output
	for (k = 0; k < 1; k++) {
		for (i = 0; i < NX; i++) {
			for (j = 0; j < NY; j++) {
				printf("%.4f %.4f\t", out[k*NX*NY+j*NX+i][0], out[k*NX*NY+j*NX+i][1]);
			}
			printf("\n");
		}
		printf("---\n");
	}
	#endif

	#endif

	#if 1 /* Cuda FFT Starts */
	float cuTime;
	cudaEvent_t custart, custop;
	cufftHandle cudaPlan;
	cufftDoubleComplex *dataCuda1, *dataCuda2;
	cufftDoubleComplex *dataLoca;
	dataLoca = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*NX*NY*NZ);
	cudaMalloc((void**)&dataCuda1, sizeof(cufftDoubleComplex)*NX*NY*NZ);
	cudaMalloc((void**)&dataCuda2, sizeof(cufftDoubleComplex)*NX*NY*NZ);
	cufftPlan3d(&cudaPlan, NX, NY, NZ, CUFFT_Z2Z);
	count = 0;	
	for (k = 0; k < NZ; k++) {
		for (j = 0; j < NY; j++) {
			for (i = 0; i < NX; i++) {
				dataLoca[k*NX*NY+j*NX+i].x = sin(2*PI*count/(NX*NY*NZ));
				dataLoca[k*NX*NY+j*NX+i].y = 0;
				count++;
			}
		}
	}

	cudaMemcpy(	dataCuda1, dataLoca, 
				sizeof(cufftDoubleComplex)*NX*NY*NZ, cudaMemcpyHostToDevice);
	cudaEventCreate(&custart);
	cudaEventCreate(&custop);
	cudaEventRecord(custart, 0);
	cufftExecZ2Z(cudaPlan, dataCuda1, dataCuda2, CUFFT_FORWARD);
	cudaEventRecord(custop, 0);
	cudaEventSynchronize(custop);
	cudaEventElapsedTime(&cuTime, custart, custop);
	cudaMemcpy(	dataLoca, dataCuda2, 
				sizeof(cufftDoubleComplex)*NX*NY*NZ, cudaMemcpyDeviceToHost);
	cufftDestroy(cudaPlan);
	cudaFree(dataCuda1);
	cudaFree(dataCuda2);
	#if 0 // Show Output
	for (k = 0; k < 1; k++) {
		for (i = 0; i < NX; i++) {
			for (j = 0; j < NY; j++) {
				printf("%g:%g\t", 	dataLoca[k*NX*NY+j*NX+i].x, 
									dataLoca[k*NX*NY+j*NX+i].y);
			}
			printf("\n");
		}
		printf("---\n");
	}
	#endif
	#endif

	#if 1
	printf("%d.%d.%d\tPRE:%f\tKER:%f\tTOT:%f\tFTW:%f\tCUD:%f\n", 	
									NX,NY,NZ, 
									((double)1.0e-9)*(plan.totalPreKernel), 
									((double)1.0e-9)*(plan.totalKernel), 
									((double)1.0e-9)*(	plan.totalKernel	+
														plan.totalPreKernel), 
									t_ns*1.0e-9, cuTime*10e-3);
	#endif
	#if 0
	plotInGnuplot(data, out, dataLoca, NX*NY*NZ);
	#endif
	#if 0
	printf("%d.%d.%d\t%f\t%f\t%f\n", 	NX,NY,NZ, 
										((double)1.0e-9)*(plan.totalKernel	+
										plan.totalPreKernel), 
										t_ns*1.0e-9, cuTime*10e-3);
	#endif
	return 0;
}


