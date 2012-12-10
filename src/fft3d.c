#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>

#ifndef TOPEFFT_H
#define TOPEFFT_H
#include "topefft.h"
#endif

void tope3DExecX(	struct topeFFT *f, 
					struct topePlan3D *t)
{
	int type = 1; // Do not Change
	f->error = clSetKernelArg(t->kernelX, 7, sizeof(int), (void*)&type);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel_swap, 7, sizeof(int), (void*)&type);
	$CHECKERROR
	
	/* Run Swapper */	
	t->globalSize[0] = t->x;
	t->globalSize[1] = t->y;
	t->globalSize[2] = t->z;
	t->localSize[0] = t->x < 128 ? t->x/2 : 128;
	t->localSize[1] = 1;
	t->localSize[2] = 1;
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_swap,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);

	/* Run Butterflies */
	t->globalSize[1] = t->y;
	t->globalSize[2] = t->z;
	t->localSize[1] = t->localSize[2] = 1;
	if (t->radX==8) {
		t->globalSize[0] = t->x/8;
		t->localSize[0] = t->x/8 < 128 ? t->x/8 : 128;
	}
	else if(t->radX==4) {
		t->globalSize[0] = t->x/4;
		t->localSize[0] = t->x/4 < 128 ? t->x/4 : 128;
	}
	else if(t->radX==2) {
		t->globalSize[0] = t->x/2;
		t->localSize[0] = t->x/2 < 128 ? t->x/2 : 128;
	}

	int s;
	for (s = 1; s <= t->logX; s++) {
		f->error = clSetKernelArg(t->kernelX, 5, sizeof(int), (void*)&s);
		$CHECKERROR

		f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernelX,
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
}

void tope3DExecY(	struct topeFFT *f, 
					struct topePlan3D *t)
{
	int type = 2; // Do not Change
	f->error = clSetKernelArg(t->kernelY, 7, sizeof(int), (void*)&type);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel_swap, 7, sizeof(int), (void*)&type);
	$CHECKERROR

	/* Run Swapper */	
	t->globalSize[0] = t->x;
	t->globalSize[1] = t->y;
	t->globalSize[2] = t->z;
	t->localSize[0] = 1;
	t->localSize[1] = t->y < 128 ? t->y/2 : 128;
	t->localSize[2] = 1;
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_swap,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);

	/* Run Butterflies */
	t->globalSize[0] = t->x;
	t->globalSize[2] = t->z;
	t->localSize[0] = t->localSize[2] = 1;
	if (t->radY==8) {
		t->globalSize[1] = t->y/8;
		t->localSize[1] = t->y/8 < 128 ? t->y/8 : 128;
	}
	else if(t->radY==4) {
		t->globalSize[1] = t->y/4;
		t->localSize[1] = t->y/4 < 128 ? t->y/4 : 128;
	}
	else if(t->radY==2) {
		t->globalSize[1] = t->y/2;
		t->localSize[1] = t->y/2 < 128 ? t->y/2 : 128;
	}

	int s;
	for (s = 1; s <= t->logY; s++) {
		f->error = clSetKernelArg(t->kernelY, 5, sizeof(int), (void*)&s);
		$CHECKERROR

		f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernelY,
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
}

void tope3DExecZ(	struct topeFFT *f, 
					struct topePlan3D *t)
{
	int type = 3; // Do not Change
	f->error = clSetKernelArg(t->kernelZ, 7, sizeof(int), (void*)&type);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel_swap, 7, sizeof(int), (void*)&type);
	$CHECKERROR

	/* Run Swapper */	
	t->globalSize[0] = t->x;
	t->globalSize[1] = t->y;
	t->globalSize[2] = t->z;
	t->localSize[0] = 1;
	t->localSize[1] = 1;
	t->localSize[2] = t->z < 128 ? t->z/2 : 128;
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_swap,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);

	/* Run Butterflies */
	t->globalSize[0] = t->x;
	t->globalSize[1] = t->y;
	t->localSize[0] = t->localSize[1] = 1;
	if (t->radZ==8) {
		t->globalSize[2] = t->z/8;
		t->localSize[2] = t->z/8 < 128 ? t->z/8 : 128;
	}
	else if(t->radZ==4) {
		t->globalSize[2] = t->z/4;
		t->localSize[2] = t->z/4 < 128 ? t->z/4 : 128;
	}
	else if(t->radZ==2) {
		t->globalSize[2] = t->z/2;
		t->localSize[2] = t->z/2 < 128 ? t->z/2 : 128;
	}

	int s;
	for (s = 1; s <= t->logZ; s++) {
		f->error = clSetKernelArg(t->kernelZ, 5, sizeof(int), (void*)&s);
		$CHECKERROR

		f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernelZ,
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
}

void tope3DExec(	struct topeFFT *f,
					struct topePlan3D *t, 
					double *d, int dir) 
{
	#if 0
	if (t->x == 2) {			// special case (not yet customized for 3D) 
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
	#endif

	/* Set Direction of Transform */
	f->error = clSetKernelArg(t->kernelX, 6, sizeof(int), (void*)&dir);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelY, 6, sizeof(int), (void*)&dir);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelZ, 6, sizeof(int), (void*)&dir);
	$CHECKERROR

	tope3DExecX(f, t);
	tope3DExecY(f, t);
	tope3DExecZ(f, t);

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

	/* Divide by N if INVERSE */
	#if 1
	if (dir == 0) {
		t->globalSize[0] = t->x;
		t->globalSize[1] = t->y;
		t->globalSize[2] = t->z;
		t->localSize[0] = t->x < 512 ? t->x/2 : 256;
		t->localSize[1] = t->localSize[2] = 1;

		f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_div,
											t->dim, NULL, t->globalSize,
											t->localSize, 0, NULL, 
											&f->event);
		$CHECKERROR
	}
	#endif

	/* Read Data Again */
	f->error = clEnqueueReadBuffer(	f->command_queue, t->data,
									CL_TRUE, 0, t->dataSize, d, 
									0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalMemory += profileThis(f->event);
}

void tope3DPlanInitBase2X(	struct topeFFT *f,
							struct topePlan3D *t)
{
	/* Twiddle Setup */
	t->twdX = clCreateBuffer(	f->context, CL_MEM_READ_WRITE,
								sizeof(double)*2*(t->x/4),
								NULL, &f->error);
	$CHECKERROR

	f->error = clSetKernelArg(	t->kernel_twid, 0, sizeof(cl_mem), 
								(void*)&t->twdX);	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_twid, 1, sizeof(int), 
								(void*)&t->x);		$CHECKERROR

	t->globalSize[0] = t->x/4;
	t->localSize[0] = t->x/4 < 512 ? t->x/4 : 256/4;
	t->globalSize[1] = t->globalSize[2] = 1;
	t->localSize[1] = t->localSize[2] = 1;
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_twid,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);
	
	/* Kernel Setup */
	switch(t->radX)
	{
		case 2:	t->kernelX = clCreateKernel(f->program3D, "DIT2C2C", &f->error);
				break;
		case 4:	t->kernelX = clCreateKernel(f->program3D, "DIT4C2C", &f->error);
				break;
		case 8:	t->kernelX = clCreateKernel(f->program3D, "DIT8C2C", &f->error);
				break;
	}
	f->error = clSetKernelArg(t->kernelX, 0, sizeof(cl_mem), (void*)&t->data);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelX, 1, sizeof(cl_mem), (void*)&t->twdX);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelX, 2, sizeof(int), (void*)&t->x);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelX, 3, sizeof(int), (void*)&t->y);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelX, 4, sizeof(int), (void*)&t->z);
	$CHECKERROR
	
	/* Bit Reversal */
	f->error = clSetKernelArg(	t->kernel_bit,0,sizeof(cl_mem), 
								(void*)&t->bitX); 	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_bit,1,sizeof(int), 
								(void*)&t->logX);	$CHECKERROR
	t->globalSize[0] = t->x/2;
	t->localSize[0] = t->x/2 < 512 ? t->x/4 : 256/2;
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_bit,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);
}

void tope3DPlanInitBase2Y(	struct topeFFT *f,
							struct topePlan3D *t)
{
	/* Twiddle Setup */
	t->twdY = clCreateBuffer(	f->context, CL_MEM_READ_WRITE,
								sizeof(double)*2*(t->y/4),
								NULL, &f->error);
	$CHECKERROR

	f->error = clSetKernelArg(	t->kernel_twid, 0, sizeof(cl_mem), 
								(void*)&t->twdY);	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_twid, 1, sizeof(int), 
								(void*)&t->y);		$CHECKERROR

	t->globalSize[0] = t->y/4;
	t->localSize[0] = t->y/4 < 512 ? t->y/4 : 256/4;
	t->globalSize[1] = t->globalSize[2] = 1;
	t->localSize[1] = t->localSize[2] = 1;
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_twid,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);

	/* Kernel Setup */
	switch(t->radY)
	{
		case 2:	t->kernelY = clCreateKernel(f->program3D, "DIT2C2C", &f->error);
				break;
		case 4:	t->kernelY = clCreateKernel(f->program3D, "DIT4C2C", &f->error);
				break;
		case 8:	t->kernelY = clCreateKernel(f->program3D, "DIT8C2C", &f->error);
				break;
	}
	f->error = clSetKernelArg(t->kernelY, 0, sizeof(cl_mem), (void*)&t->data);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelY, 1, sizeof(cl_mem), (void*)&t->twdY);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelY, 2, sizeof(int), (void*)&t->x);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelY, 3, sizeof(int), (void*)&t->y);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelY, 4, sizeof(int), (void*)&t->z);
	$CHECKERROR
	
	/* Bit Reversal */
	f->error = clSetKernelArg(	t->kernel_bit,0,sizeof(cl_mem), 
								(void*)&t->bitY); 	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_bit,1,sizeof(int), 
								(void*)&t->logY);	$CHECKERROR
	t->globalSize[0] = t->y/2;
	t->localSize[0] = t->y/2 < 512 ? t->y/4 : 256/2;
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_bit,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);
}

void tope3DPlanInitBase2Z(	struct topeFFT *f,
							struct topePlan3D *t)
{
	/* Twiddle Setup */
	t->twdZ = clCreateBuffer(	f->context, CL_MEM_READ_WRITE,
								sizeof(double)*2*(t->z/4),
								NULL, &f->error);
	$CHECKERROR

	f->error = clSetKernelArg(	t->kernel_twid, 0, sizeof(cl_mem), 
								(void*)&t->twdZ);	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_twid, 1, sizeof(int), 
								(void*)&t->z);		$CHECKERROR

	t->globalSize[0] = t->z/4;
	t->localSize[0] = t->z/4 < 512 ? t->z/4 : 256/4;
	t->globalSize[1] = t->globalSize[2] = 1;
	t->localSize[1] = t->localSize[2] = 1;
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_twid,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);

	/* Kernel Setup */
	switch(t->radZ)
	{
		case 2:	t->kernelZ = clCreateKernel(f->program3D, "DIT2C2C", &f->error);
				break;
		case 4:	t->kernelZ = clCreateKernel(f->program3D, "DIT4C2C", &f->error);
				break;
		case 8:	t->kernelZ = clCreateKernel(f->program3D, "DIT8C2C", &f->error);
				break;
	}
	f->error = clSetKernelArg(t->kernelZ, 0, sizeof(cl_mem), (void*)&t->data);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelZ, 1, sizeof(cl_mem), (void*)&t->twdZ);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelZ, 2, sizeof(int), (void*)&t->x);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelZ, 3, sizeof(int), (void*)&t->y);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernelZ, 4, sizeof(int), (void*)&t->z);
	$CHECKERROR
	
	/* Bit Reversal */
	f->error = clSetKernelArg(	t->kernel_bit,0,sizeof(cl_mem), 
								(void*)&t->bitZ); 	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_bit,1,sizeof(int), 
								(void*)&t->logZ);	$CHECKERROR
	t->globalSize[0] = t->z/2;
	t->localSize[0] = t->z/2 < 512 ? t->z/4 : 256/2;
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_bit,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);
}

void tope3DPlanInit(struct topeFFT *f, 
					struct topePlan3D *t, 
					int x, int y, int z, int type, double *d) 
{
	/* Some Simple Initializations */
	t->totalMemory = t->totalKernel = t->totalPreKernel = 0;
	t->x = x;			// size
	t->y = y;
	t->z = z;
	t->logX = log2(x);	// Log
	t->logY = log2(y);
	t->logZ = log2(z);
	t->type = type;		// C2C/R2C etc
	t->dim = 3;			// Dimensions for kernel
	t->globalSize = malloc(sizeof(size_t)*t->dim);	// Kernel indexing
	t->localSize = malloc(sizeof(size_t)*t->dim);

	/* Decide Radix */
	if( t->logX % 3==0 ) t->radX = 8;
	else{	
		if ( (t->logX) % 2 == 0) t->radX = 4;
		else {
			if (x % 2 == 0) t->radX = 2;
			else {
				t->radX = -1;
			}
		}
	}
	if( t->logY % 3==0 ) t->radY = 8;
	else{	
		if ( (t->logY) % 2 == 0) t->radY = 4;
		else {
			if (y % 2 == 0) t->radY = 2;
			else {
				t->radY = -1;
			}
		}
	}
	if( t->logZ % 3==0 ) t->radZ = 8;
	else{	
		if ( (t->logZ) % 2 == 0) t->radZ = 4;
		else {
			if (z % 2 == 0) t->radZ = 2;
			else {
				t->radZ = -1;
			}
		}
	}
	if (t->radX == -1) {
		printf("No algorithm for x input size %d\n", t->x);
	 	exit(0);
	}
	if (t->radY == -1) {
		printf("No algorithm for y input size %d\n", t->y);
	 	exit(0);
	}
	if (t->radZ == -1) {
		printf("No algorithm for z input size %d\n", t->z);
	 	exit(0);
	}

	/* Memory Allocation */
	t->dataSize = sizeof(double)*2*x*y*z;
	t->data   = clCreateBuffer(	f->context, CL_MEM_READ_WRITE,
								t->dataSize, NULL, &f->error);
	$CHECKERROR
	t->bitX = clCreateBuffer(	f->context, CL_MEM_READ_WRITE,
								sizeof(int)*x, NULL, &f->error);
	$CHECKERROR
	t->bitY = clCreateBuffer(	f->context, CL_MEM_READ_WRITE,
								sizeof(int)*y, NULL, &f->error);
	$CHECKERROR
	t->bitZ = clCreateBuffer(	f->context, CL_MEM_READ_WRITE,
								sizeof(int)*z, NULL, &f->error);
	$CHECKERROR

	/* Swapping Kernel Setup */
	t->kernel_swap = clCreateKernel(f->program3D, "swapkernel", &f->error);
	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_swap,	0,
								sizeof(cl_mem), (void*)&t->data); $CHECKERROR
	f->error = clSetKernelArg(	t->kernel_swap,	1,
								sizeof(int), (void*)&t->x); $CHECKERROR
	f->error = clSetKernelArg(	t->kernel_swap,	2,
								sizeof(int), (void*)&t->y); $CHECKERROR
	f->error = clSetKernelArg(	t->kernel_swap,	3,
								sizeof(int), (void*)&t->z); $CHECKERROR
	f->error = clSetKernelArg(	t->kernel_swap,	4,
								sizeof(cl_mem),	(void*)&t->bitX); $CHECKERROR
	f->error = clSetKernelArg(	t->kernel_swap,	5,
								sizeof(cl_mem),	(void*)&t->bitY); $CHECKERROR
	f->error = clSetKernelArg(	t->kernel_swap,	6,
								sizeof(cl_mem),	(void*)&t->bitZ); $CHECKERROR

	/* Divide Kernel for Inverse */
	t->kernel_div = clCreateKernel( f->program3D, "divide", &f->error);
	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_div,0,sizeof(cl_mem),
								(void*)&t->data); $CHECKERROR
	f->error = clSetKernelArg(	t->kernel_div,1,sizeof(int),
								(void*)&t->x); $CHECKERROR
	f->error = clSetKernelArg(	t->kernel_div,2,sizeof(int),
								(void*)&t->y); $CHECKERROR
	f->error = clSetKernelArg(	t->kernel_div,3,sizeof(int),
								(void*)&t->z); $CHECKERROR

	/* Create Twid & Bit Kernel. Arguments set in next section */
	t->kernel_twid = clCreateKernel(f->program3D, "twiddles", &f->error);
	$CHECKERROR
	t->kernel_bit = clCreateKernel(	f->program3D, "reverse", &f->error);
	$CHECKERROR

	/* Send Rest of Setup to Right Function s*/
	if ((t->radX == 2 || t->radX == 4) || t->radX == 8) {
		tope3DPlanInitBase2X(f,t);
	}
	if ((t->radY == 2 || t->radY == 4) || t->radY == 8) {
		tope3DPlanInitBase2Y(f,t);
	}
	if ((t->radZ == 2 || t->radZ == 4) || t->radZ == 8) {
		tope3DPlanInitBase2Z(f,t);
	}
	
	#if 0 // Debug Code
	int i;
	int test[t->x];	
	f->error = clEnqueueReadBuffer(	f->command_queue, t->bitZ,
									CL_TRUE, 0, sizeof(int)*t->x, test, 
									0, NULL, &f->event);
	for (i = 0; i < t->x; i++) {
		//printf("%lf:%lf\n", d[2*i], d[2*i+1]);
		printf("%d\n", test[i]);
	}
	#endif

	/* Write Data */
	f->error = clEnqueueWriteBuffer(f->command_queue, t->data,
									CL_TRUE, 0, t->dataSize, d, 
									0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalMemory += profileThis(f->event);

	/* Readjustments */
	if(t->radX==8)		t->logX=t->logX/3;
	else if(t->radX==4)	t->logX=t->logX/2;
	
	if(t->radY==8)		t->logY=t->logY/3;
	else if(t->radY==4)	t->logY=t->logY/2;

	if(t->radZ==8)		t->logZ=t->logZ/3;
	else if(t->radZ==4)	t->logZ=t->logZ/2;
}

void tope3DDestroy(	struct topeFFT *f,
					struct topePlan3D *t) 
{
	f->error = clFlush(f->command_queue);
	f->error = clFinish(f->command_queue);
	f->error = clReleaseKernel(t->kernelX);
	f->error = clReleaseKernel(t->kernelY);
	f->error = clReleaseKernel(t->kernelZ);
	f->error = clReleaseKernel(t->kernel_bit);
	f->error = clReleaseKernel(t->kernel_swap);
	f->error = clReleaseKernel(t->kernel_twid);
	f->error = clReleaseKernel(t->kernel_div);
	f->error = clReleaseProgram(f->program1D);
	//f->error = clReleaseProgram(f->program2D);
	f->error = clReleaseProgram(f->program3D);
	f->error = clReleaseMemObject(t->data);
	f->error = clReleaseMemObject(t->bitX);
	f->error = clReleaseMemObject(t->bitY);
	f->error = clReleaseMemObject(t->bitZ);
	f->error = clReleaseMemObject(t->twdX);
	f->error = clReleaseMemObject(t->twdY);
	f->error = clReleaseMemObject(t->twdZ);
	f->error = clReleaseCommandQueue(f->command_queue);
	f->error = clReleaseContext(f->context);
}

