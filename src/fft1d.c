#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>

#ifndef TOPEFFT_H
#define TOPEFFT_H
#include "topefft.h"
#endif

void tope1DExec(	struct topeFFT *f,
					struct topePlan1D *t, 
					double *d, int dir) 
{
	if (t->x == 2) {			// special case size 2 input
		double hr = d[0];
		double hi = d[1];
		d[0] += d[2];
		d[1] += d[3];
		d[2] = hr - d[2];
		d[3] = hi - d[3];
		if (dir == 0) {
			d[0] /= t->x;
			d[2] /= t->x;
		}
		return;
	}

	/* Set Direction of Transform */
	f->error = clSetKernelArg(t->kernel, 4, sizeof(int), (void*)&dir);
	$CHECKERROR

	/* Run Swapper */	
	f->error = clSetKernelArg(t->kernel_swap,0,sizeof(cl_mem), (void*)&t->data);
	$CHECKERROR

	t->globalSize[0] = t->x;
	t->localSize[0] = t->x < 128 ? t->x/2 : 128;
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_swap,
											t->dim, NULL, t->globalSize,
											t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);

	#if 0 // Debug Code
	int i;
	f->error = clEnqueueReadBuffer(	f->command_queue, t->data,
									CL_TRUE, 0, t->dataSize, d, 
									0, NULL, &f->event);
	for (i = 0; i < t->x; i++) {
		printf("%f:%f\n", d[2*i], d[2*i+1]);	
	}
	exit(0);
	#endif

	/* Run Butterflies */
	if(t->radix==8) {
		t->globalSize[0] = (t->x)/8;
		t->localSize[0] = ((t->x)/8) < 128 ? (t->x)/8 : 128;
	}
	else {
		if(t->radix==4) {
			t->globalSize[0] = (t->x)/4;
			t->localSize[0] = ((t->x)/4) < 128 ? (t->x)/4 : 128;
		}
		else { 
			if(t->radix==2){
				t->globalSize[0] = (t->x)/2;
				t->localSize[0] = ((t->x)/2) < 128 ? (t->x)/2 : 128;
			}
		}
	}

	int x;
	for (x = 1; x <= t->log; x++) {
		f->error = clSetKernelArg(t->kernel, 3, sizeof(int), (void*)&x);
		$CHECKERROR

		f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel,
											t->dim, NULL, t->globalSize,
											t->localSize, 0, NULL, &f->event);
		$CHECKERROR
		clFinish(f->command_queue);
		t->totalKernel += profileThis(f->event);

		#if 0 // Debug Code
			int i;
			f->error = clEnqueueReadBuffer(	f->command_queue, t->data,
										CL_TRUE, 0, t->dataSize, d, 
										0, NULL, &f->event);
			$CHECKERROR
			for (i = 0; i < t->x; i++) {
				printf("%lf:%lf\n", d[2*i], d[2*i+1]);
			}
		#endif
	}

	/* Divide by N if INVERSE */
	if (dir == 0) {
		t->globalSize[0] = t->x;
		t->localSize[0] = t->x < 512 ? t->x/2 : 256;

		f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_div,
											t->dim, NULL, t->globalSize,
											t->localSize, 0, NULL, 
											&f->event);
		$CHECKERROR
	}

	/* Read Data Again */
	f->error = clEnqueueReadBuffer(	f->command_queue, t->data,
									CL_TRUE, 0, t->dataSize, d, 
									0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalMemory += profileThis(f->event);
}

void tope1DPlanInitBase2(	struct topeFFT *f,
							struct topePlan1D *t, int x)
{
	/* Twiddle Setup */
	t->twiddle = clCreateBuffer(f->context, CL_MEM_READ_WRITE,
								sizeof(double)*2*(x/4),
								NULL, &f->error);
	$CHECKERROR

	t->kernel_twid = clCreateKernel(f->program1D, "twid1D", &f->error);
	$CHECKERROR

	f->error = clSetKernelArg(	t->kernel_twid, 0, sizeof(cl_mem), 
								(void*)&t->twiddle);	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_twid, 1, sizeof(int), 
								(void*)&x);				$CHECKERROR

	t->globalSize[0] = x/4;
	t->localSize[0] = x/4 < 512 ? x/4 : 256/4;
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_twid,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);

	/* Kernel Setup */
	switch(t->radix)
	{
		case 2:	t->kernel = clCreateKernel(f->program1D, "DIT2C2C", &f->error);
				break;
		case 4:	t->kernel = clCreateKernel(f->program1D, "DIT4C2C", &f->error);
				break;
		case 8:	t->kernel = clCreateKernel(f->program1D, "DIT8C2C", &f->error);
				break;
	}
	f->error = clSetKernelArg(t->kernel, 0, sizeof(cl_mem), (void*)&t->data);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel, 1, sizeof(cl_mem), (void*)&t->twiddle);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel, 2, sizeof(int), (void*)&t->x);
	$CHECKERROR

	/* Divide Kernel for Inverse */
	t->kernel_div = clCreateKernel( f->program1D, "divide1D", &f->error);
	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_div,0,sizeof(cl_mem),
								(void*)&t->data); $CHECKERROR
	f->error = clSetKernelArg(	t->kernel_div,1,sizeof(int),
								(void*)&t->x); $CHECKERROR
	
	/* Bit Reversal */
	t->kernel_bit = clCreateKernel(	f->program1D, "reverse2", &f->error);
	$CHECKERROR
	
	f->error = clSetKernelArg(	t->kernel_bit,0,sizeof(cl_mem), 
								(void*)&t->bitrev); $CHECKERROR
	f->error = clSetKernelArg(	t->kernel_bit,1,sizeof(int), 
								(void*)&t->log);	$CHECKERROR
	t->globalSize[0] = x/2;
	t->localSize[0] = x/2 < 512 ? x/4 : 256/2;
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_bit,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);
}

void tope1DPlanInit(struct topeFFT *f, 
					struct topePlan1D *t, 
					int x, int type, double *d) 
{
	/* Some Simple Initializations */
	t->totalMemory = t->totalKernel = t->totalPreKernel = 0;
	t->x = x;			// size
	t->log = log2(x);	// Log
	t->type = type;		// C2C/R2C etc
	t->dim = 1;			// Dimensions for kernel
	t->globalSize = malloc(sizeof(size_t)*t->dim);	// Kernel indexing
	t->localSize = malloc(sizeof(size_t)*t->dim);

	/* Decide Radix */
	if( t->log % 3==0 ) t->radix = 8;
	else{	
		if ( (t->log) % 2 == 0) t->radix = 4;
		else {
			if (x % 2 == 0) t->radix = 2;
			else {
				t->radix = -1;
			}
		}
	}
	if (t->radix == -1) {
		printf("No algorithm for this input size\n");
	 	exit(0);
	}

	/* Memory Allocation */
	t->dataSize = sizeof(double)*x*2;
	t->data   = clCreateBuffer(	f->context, CL_MEM_READ_WRITE,
								t->dataSize, NULL, &f->error);
	$CHECKERROR
	t->bitrev = clCreateBuffer(	f->context, CL_MEM_READ_WRITE,
								sizeof(int)*x, NULL, &f->error);
	$CHECKERROR

	/* Swapping Kernel Setup*/
	t->kernel_swap = clCreateKernel(f->program1D, "swap1D", &f->error);
	clCreateKernelChecker(&f->error);
	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_swap,1,sizeof(cl_mem),
								(void*)&t->bitrev); $CHECKERROR

	/* Send Rest of Setup to Right Function s*/
	if ((t->radix == 2 || t->radix == 4) || t->radix == 8) {
		if (x > 2) tope1DPlanInitBase2(f,t,x);
	}

	/* Write Data */
	f->error = clEnqueueWriteBuffer(f->command_queue, t->data,
									CL_TRUE, 0, t->dataSize, d, 
									0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalMemory += profileThis(f->event);

	/* Readjustments */
	if(t->radix==8)
		t->log=t->log/3;
	if(t->radix==4)
		t->log=t->log/2;
}

void tope1DDestroy(	struct topeFFT *f,
					struct topePlan1D *t) 
{
	if (t->x > 2) // Not initialized for under 2 
	{
		f->error = clFlush(f->command_queue);
		f->error = clFinish(f->command_queue);
		f->error = clReleaseKernel(t->kernel);
		f->error = clReleaseKernel(t->kernel_bit);
		f->error = clReleaseKernel(t->kernel_swap);
		f->error = clReleaseKernel(t->kernel_twid);
		f->error = clReleaseKernel(t->kernel_div);
		f->error = clReleaseProgram(f->program1D);
		f->error = clReleaseProgram(f->program2D);
		f->error = clReleaseProgram(f->program3D);
		f->error = clReleaseMemObject(t->data);
		f->error = clReleaseMemObject(t->bitrev);
		f->error = clReleaseMemObject(t->twiddle);
		f->error = clReleaseCommandQueue(f->command_queue);
		f->error = clReleaseContext(f->context);
	}
}

