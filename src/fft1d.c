#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>

#ifndef TOPEFFT_H
#define TOPEFFT_H
#include "topefft.h"
#endif
#include "checkers.h"

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

	if (t->radix > 0) {
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

		printf("%d\n", t->radix);

		$CHECKERROR
		clFinish(f->command_queue);
		t->totalPreKernel += profileThis(f->event);
	}
	else {
		/* Set Direction of Transform */
		f->error = clSetKernelArg(t->kernel, 3, sizeof(int), (void*)&dir);
		$CHECKERROR
	}

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
	else if(t->radix==4) {
		t->globalSize[0] = (t->x)/4;
		t->localSize[0] = ((t->x)/4) < 128 ? (t->x)/4 : 128;
	}
	else if(t->radix==2){
		t->globalSize[0] = (t->x)/2;
		t->localSize[0] = ((t->x)/2) < 128 ? (t->x)/2 : 128;
	}
	else if (t->radix == -1) {
		t->globalSize[0] = t->x;
		t->localSize[0] = t->x < 512 ? t->x : t->x % 2 == 0 ? 2 : 1;
	}

	int x;

	switch(t->radix)
	{
		case 8:
		case 4:
		case 2:
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
		break;
		case -1: // DFT Case
			f->error = clEnqueueNDRangeKernel(	
									f->command_queue, t->kernel,
									t->dim, NULL, t->globalSize,
									t->localSize, 0, NULL, &f->event);
									$CHECKERROR
			clFinish(f->command_queue);
			t->totalKernel += profileThis(f->event);
		break;
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
	f->error = clEnqueueReadBuffer(	f->command_queue, t->scratch,
									CL_TRUE, 0, t->dataSize, d, 
									0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalMemory += profileThis(f->event);
}

void tope1DPlanInitDFT( 	struct topeFFT *f,
							struct topePlan1D *t, int x)
{
	/* Twiddle Setup 
	 * -------------
	 * Dev: Note:
	 * ---------
	 * The size of the twiddle input is +1'd to allow calculation of the last 
	 * twiddle. Eventually, the twiddle calculations would be /4 and recovered 
	 * using the quad/buad technique from the radix kernels in the code.
	 * For example, the last twiddle for input size 5 would be W^16.
	 *
	 * Also, if twiddles are calculated on the fly, the twiddle code in this
	 * section is redundant.
	 */

	#if 0
	int required = (x-1) * (x-1) +1; // Eventually reduce/4 !!!!!

	t->twiddle = clCreateBuffer(f->context, CL_MEM_READ_WRITE,
								sizeof(double)*2*required,
								NULL, &f->error);
	$CHECKERROR

	t->kernel_twid = clCreateKernel(f->program1D, "twid1D", &f->error);
	$CHECKERROR

	f->error = clSetKernelArg(	t->kernel_twid, 0, sizeof(cl_mem), 
								(void*)&t->twiddle);	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_twid, 1, sizeof(int), 
								(void*)&t->x);		$CHECKERROR

	t->globalSize[0] = 512;//required;
	if (required % 2 == 0) {
		t->localSize[0] = 32;//required < 512 ? required : 2; 
	}
	else {
		t->localSize[0] = 1;//required < 512 ? required : 1; 
	}
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_twid,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);
	#endif

	#if 0 // Debug Code
	int i,j;
	double d[required*2];
	f->error = clEnqueueReadBuffer(	f->command_queue, t->twiddle,
									CL_TRUE, 0, sizeof(double)*2*required, d, 
									0, NULL, &f->event);
	for (i = 0; i < t->x; i++) {
		for (j = 0; j < t->x; j++) {
			printf("%f\t%f\t%d\n", d[(2*i*j)], d[(2*i*j)+1], i*j);	
		}
	}
	//exit(0);
	#endif

	/* Create Scratch Space
	 * ---------------------
	 *  This space is required because OpenCL is giving problems in long roops
	 *  and therefore in order to break up the loops into chunks, we need to
	 *  store the multiplye-and-adds in temporary storge (so the original data
	 *  is not touched)
	 *
	 *  Update: It had nothing to do with long loops !!!! hahaha
	 */
	t->scratch = clCreateBuffer(f->context, CL_MEM_READ_WRITE,
								sizeof(double)*2*t->x,
								NULL, &f->error); 			$CHECKERROR

	/* Kernel Setup */
	t->kernel = clCreateKernel(f->program1D, "DFT", &f->error);
	$CHECKERROR

	f->error = clSetKernelArg(t->kernel, 0, sizeof(cl_mem), (void*)&t->data);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel, 1, sizeof(cl_mem), (void*)&t->scratch);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel, 2, sizeof(int), (void*)&t->x);
	$CHECKERROR
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
	t->type = type;		// C2C/R2C etc
	t->dim = 1;			// Dimensions for kernel
	t->globalSize = malloc(sizeof(size_t)*t->dim);	// Kernel indexing
	t->localSize = malloc(sizeof(size_t)*t->dim);

	/* Separates integer and decimal parts */
	double re;
	double fl = modf(log2(x),&re); 

	/* Decide Radix */
	if (!fl) {
		t->log = log2(x);	// Log
		if( t->log % 3==0 ) t->radix = 8;
		else if ( (t->log) % 2 == 0) t->radix = 4;
		else if (x % 2 == 0) t->radix = 2;
		else t->radix = -1;
 	}
	else {
		t->radix = -1;
	}

	/* Memory Allocation */
	t->dataSize = sizeof(double)*x*2;
	t->data   = clCreateBuffer(	f->context, CL_MEM_READ_WRITE,
								t->dataSize, NULL, &f->error);
	$CHECKERROR
	t->bitrev = clCreateBuffer(	f->context, CL_MEM_READ_WRITE,
								sizeof(int)*x, NULL, &f->error);
	$CHECKERROR

	if (t->radix > 0) {
	/* Swapping Kernel Setup */
		t->kernel_swap = clCreateKernel(f->program1D, "swap1D", &f->error);
		clCreateKernelChecker(&f->error);
		$CHECKERROR
		f->error = clSetKernelArg(	t->kernel_swap,1,sizeof(cl_mem),
									(void*)&t->bitrev); $CHECKERROR
	}

	/* Send Rest of Setup to Right Function s*/
	if ((t->radix == 2 || t->radix == 4) || t->radix == 8) {
		if (x > 2) tope1DPlanInitBase2(f,t,x);
	}
	if (t->radix == -1) {
		tope1DPlanInitDFT(f,t,x);
	}

	/* Write Data */
	f->error = clEnqueueWriteBuffer(f->command_queue, t->data,
									CL_TRUE, 0, t->dataSize, d, 
									0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalMemory += profileThis(f->event);

	/* Readjustments */
	if (t->radix > 0) {
		if(t->radix==8)			t->log=t->log/3;
		if(t->radix==4)			t->log=t->log/2;
	}
}

void tope1DDestroy(	struct topeFFT *f,
					struct topePlan1D *t) 
{
	if (t->x > 2) // Not initialized for under 2 
	{
		if (t->radix > 0) {
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
		else {
			f->error = clFlush(f->command_queue);
			f->error = clFinish(f->command_queue);
			f->error = clReleaseKernel(t->kernel);
			f->error = clReleaseProgram(f->program1D);
			f->error = clReleaseProgram(f->program2D);
			f->error = clReleaseProgram(f->program3D);
			f->error = clReleaseMemObject(t->data);
			f->error = clReleaseMemObject(t->scratch);
			f->error = clReleaseCommandQueue(f->command_queue);
			f->error = clReleaseContext(f->context);
		}
	}
}

