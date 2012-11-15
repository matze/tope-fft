#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>

#ifndef TOPEFFT_H
#define TOPEFFT_H
#include "topefft.h"
#endif

#include <fftw3.h>

void plotInGnuplot(double *d, fftw_complex *o, int n) {
	FILE *gplot = popen("gnuplot -persistent", "w");
	FILE *temp = fopen("delete", "w");
	int i;
	#if 0
	#if 1
	fprintf(gplot, "plot '-' title 'topeFFT' w l \n");
	for (i = 0; i < n; i++)  
		fprintf(gplot, "%lf\n", pow(pow(d[2*i],2)+pow(d[2*i+1],2),0.5));
	fprintf(gplot, "e");
	#endif
	fprintf(gplot, "plot '-' title 'FFTW' w l \n");
	for (i = 0; i < n; i++)  
		fprintf(gplot, "%lf\n", pow(pow(o[i][0],2)+pow(o[i][1],2),0.5));
	fprintf(gplot, "e");
	#endif
	#if 1
	for (i = 0; i < n; i++) { 
		fprintf(temp, "%d\t%lf\t%lf\n", 
						i, 
						pow(pow(d[2*i],2)+pow(d[2*i+1],2),0.5),
						pow(pow(o[i][0],2)+pow(o[i][1],2),0.5));
	}
	fprintf(gplot, 	"plot 'delete' using 1:2 title 'topeFFT' w l lw 4,"\
					"'delete' using 1:3 title 'FFTW' w l lw 2\n");
	#endif
}

cl_ulong profileThis(cl_event a) {
	cl_ulong start = 0, end = 0;
	clGetEventProfilingInfo(a, CL_PROFILING_COMMAND_END, 
							sizeof(cl_ulong), &end, NULL);
	clGetEventProfilingInfo(a, CL_PROFILING_COMMAND_START, 
							sizeof(cl_ulong), &start, NULL);
	return (end - start);
}

void tope1DExec(	struct topeFFT *f,
					struct topePlan1D *t, 
					double *d, int dir) 
{
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
	t->globalSize[0] = (t->x)/2;
	t->localSize[0] = ((t->x)/2) < 128 ? (t->x)/2 : 128;
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
	}

	/* Read Data Again */
	f->error = clEnqueueReadBuffer(	f->command_queue, t->data,
									CL_TRUE, 0, t->dataSize, d, 
									0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalMemory += profileThis(f->event);

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
	t->kernel_swap = clCreateKernel(f->program, "swap1D", &f->error);
	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_swap,1,sizeof(cl_mem),
								(void*)&t->bitrev); $CHECKERROR

	/* Twiddle Setup */
	t->twiddle = clCreateBuffer(f->context, CL_MEM_READ_WRITE,
								sizeof(double)*2*(x/4),
								NULL, &f->error);
	$CHECKERROR

	t->kernel_twid = clCreateKernel(f->program, "twid1D", &f->error);
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
		case 2:	t->kernel = clCreateKernel(f->program, "DIT2C2C", &f->error);
				break;
		case 4:	t->kernel = clCreateKernel(f->program, "DIT4C2C", &f->error);
				break;

	}
	f->error = clSetKernelArg(t->kernel, 0, sizeof(cl_mem), (void*)&t->data);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel, 1, sizeof(cl_mem), (void*)&t->twiddle);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel, 2, sizeof(int), (void*)&t->x);
	$CHECKERROR
	
	/* Bit Reversal */
	t->kernel_bit = clCreateKernel(	f->program, "reverse2", &f->error);
	$CHECKERROR
	
	f->error = clSetKernelArg(	t->kernel_bit,0,sizeof(cl_mem), 
								(void*)&t->bitrev); $CHECKERROR
	f->error = clSetKernelArg(	t->kernel_bit,1,sizeof(int), 
								(void*)&t->log);	$CHECKERROR
	t->globalSize[0] = x;
	t->localSize[0] = x < 512 ? x/2 : 256;
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_bit,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);

	f->error = clEnqueueWriteBuffer(f->command_queue, t->data,
								CL_TRUE, 0, t->dataSize, d, 
								0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalMemory += profileThis(f->event);
	if(t->radix==4)
		t->log=t->log/2;
}

void topeFFTInit(struct topeFFT *f)
{
	// Platform
	f->error = clGetPlatformIDs(1, &f->platform_id, &f->ret_num_platforms);
	$CHECKERROR

	// Device
	f->error = clGetDeviceIDs(	f->platform_id, 
								CL_DEVICE_TYPE_DEFAULT, 1, &f->device, 
								&f->ret_num_devices);
	$CHECKERROR

	// Some Info
	size_t info_size;
	cl_char info[1024];
	cl_ulong info_num;
	f->error = clGetDeviceInfo(	f->device, CL_DEVICE_VENDOR, sizeof(info), 
								info, &info_size);
	fprintf(stderr,"Device to Use: %s\n", info);

	f->error = clGetDeviceInfo(	f->device, CL_DEVICE_NAME, sizeof(info), 
								info, &info_size);
	fprintf(stderr,"%s\n", info);

	f->error = clGetDeviceInfo(	f->device, CL_DEVICE_GLOBAL_MEM_SIZE, 
								sizeof(info_num), &info_num, &info_size);
	fprintf(stderr,"Global Memory: %f Mb\n", (float)(info_num/1024/1024));
	f->error = clGetDeviceInfo(f->device, CL_DEVICE_LOCAL_MEM_SIZE, 
								sizeof(info_num), &info_num, &info_size);
	fprintf(stderr,"Local Memory Size: %f Kb\n", (float)(info_num/1024));
	
	f->error = clGetDeviceInfo(	f->device, 
								CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, 
								sizeof(info), &info, &info_size);
	fprintf(stderr,"double Support: %s\n", info == 0 ? "no" : "yes"); 

	// Context
	f->context = clCreateContext(NULL, 1, &f->device, NULL, NULL, &f->error);
	$CHECKERROR

	// Command Queue 	
	f->command_queue = clCreateCommandQueue(f->context, f->device, 
											CL_QUEUE_PROFILING_ENABLE, 
											&f->error);
	$CHECKERROR

	// CL File		
	FILE *fp; 
	char *source_str;
	size_t source_size;
	char fileName[] = "src/kernels.cl";
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernels. Check path.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	f->program = clCreateProgramWithSource(	f->context, 1, 
											(const char **)&source_str, 
											(const size_t *)&source_size, 
											&f->error);
	$CHECKERROR

	// Build
	char buffer[2048];
	f->error = clBuildProgram(	f->program, 1, &f->device, 
								"-cl-nv-verbose", NULL, NULL);
	clGetProgramBuildInfo(	f->program, f->device, CL_PROGRAM_BUILD_LOG, 
							sizeof(buffer), buffer, NULL);
	size_t ret_value_size;
	clGetProgramBuildInfo(	f->program, f->device, CL_PROGRAM_BUILD_LOG, 0, 
							NULL, &ret_value_size);
	char *register1 = malloc(ret_value_size+1);
	clGetProgramBuildInfo(	f->program, f->device, CL_PROGRAM_BUILD_LOG, 
							ret_value_size, register1, NULL);
	register1[ret_value_size] = '\0';
	fprintf(stderr, "%s\n", register1);

	#if 0 // Get PTX Yes/No
    size_t *binary_sizes = (size_t*)malloc(t->ret_num_devices * sizeof(size_t));    
    clGetProgramInfo(	t->program, CL_PROGRAM_BINARY_SIZES, 
						t->ret_num_devices * sizeof(size_t), binary_sizes, NULL);
    char** ptx_code = (char**) malloc(t->ret_num_devices * sizeof(char*));
	int i;
    for(i=0; i<t->ret_num_devices; ++i) ptx_code[i]= (char*)malloc(binary_sizes[i]);
    clGetProgramInfo(t->program, CL_PROGRAM_BINARIES, 0, ptx_code, NULL);
	printf("%s\n", ptx_code[0]);
	#endif
}

