#include <stdio.h>

#ifndef TOPEFFT_H
#define TOPEFFT_H
#include "topefft.h"                                                            
#endif 

int main() {

	struct topeFFT f;

	// Platform
	f.error = clGetPlatformIDs(1, &f.platform_id, &f.ret_num_platforms);

	// Device
	f.error = clGetDeviceIDs(	f.platform_id, 
								CL_DEVICE_TYPE_DEFAULT, 1, &f.device, 
								&f.ret_num_devices);
	// Some Info
	size_t info_size;
	cl_char info[1024];
	f.error = clGetDeviceInfo(	f.device, CL_DEVICE_VENDOR, sizeof(info), 
								info, &info_size);
	fprintf(stderr,"Device to Use: %s\n", info);

	f.error = clGetDeviceInfo(	f.device, CL_DEVICE_NAME, sizeof(info), 
								info, &info_size);
	fprintf(stderr,"%s\n", info);

	// Context
	f.context = clCreateContext(NULL, 1, &f.device, NULL, NULL, &f.error);

	// Command Queue 	
	f.command_queue = clCreateCommandQueue(f.context, f.device, 
											CL_QUEUE_PROFILING_ENABLE, 
											&f.error);

	#if 1 /* Prepare 1D Program */
	// CL File		
	FILE *fp; 
	char *source_str;
	size_t source_size;
	char fileName[] = "src/kernels1D.cl";
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernels. Check path.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	f.program1D = clCreateProgramWithSource(	f.context, 1, 
												(const char **)&source_str, 
												(const size_t *)&source_size, 
												&f.error);

	// Build
	char buffer[2048];
	f.error = clBuildProgram(	f.program1D, 1, &f.device, 
								"-cl-nv-verbose", NULL, NULL);
	clGetProgramBuildInfo(	f.program1D, f.device, CL_PROGRAM_BUILD_LOG, 
							sizeof(buffer), buffer, NULL);
	size_t ret_value_size;
	clGetProgramBuildInfo(	f.program1D, f.device, CL_PROGRAM_BUILD_LOG, 0, 
							NULL, &ret_value_size);
	char *register1 = malloc(ret_value_size+1);
	clGetProgramBuildInfo(	f.program1D, f.device, CL_PROGRAM_BUILD_LOG, 
							ret_value_size, register1, NULL);
	register1[ret_value_size] = '\0';
	fprintf(stderr, "%s\n", register1);

	FILE *file1D = fopen("src/kernels1D.ptx", "w");
    size_t *binary_sizes = (size_t*)malloc(f.ret_num_devices * sizeof(size_t) - 5 );
    clGetProgramInfo(f.program1D, CL_PROGRAM_BINARY_SIZES, f.ret_num_devices * sizeof(size_t), binary_sizes, NULL);
    char** ptx_code = (char**) malloc(f.ret_num_devices * sizeof(char*));
	int i;
    for(i=0; i<f.ret_num_devices; ++i) {
        ptx_code[i]= (char*)malloc(binary_sizes[i]);
    }
    clGetProgramInfo(f.program1D, CL_PROGRAM_BINARIES, 0, ptx_code, NULL);
	fprintf(file1D, "%s\n", ptx_code[0]);
	#endif

	#if 1 /* Prepare 2D Program */
	char file2D[] = "src/kernels2D.cl";
	fp = fopen(file2D, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernels. Check path.\n");
		exit(1);
	}
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	f.program2D = clCreateProgramWithSource(	f.context, 1, 
												(const char **)&source_str, 
												(const size_t *)&source_size, 
												&f.error);

	// Build
	f.error = clBuildProgram(	f.program2D, 1, &f.device, 
								"-cl-nv-verbose", NULL, NULL);
	clGetProgramBuildInfo(	f.program2D, f.device, CL_PROGRAM_BUILD_LOG, 
							sizeof(buffer), buffer, NULL);
	clGetProgramBuildInfo(	f.program2D, f.device, CL_PROGRAM_BUILD_LOG, 0, 
							NULL, &ret_value_size);
	register1 = realloc(register1, ret_value_size+1);
	clGetProgramBuildInfo(	f.program2D, f.device, CL_PROGRAM_BUILD_LOG, 
							ret_value_size, register1, NULL);
	register1[ret_value_size] = '\0';
	fprintf(stderr, "%s\n", register1);

	file1D = fopen("src/kernels2D.ptx", "w");
    binary_sizes = (size_t*)malloc(f.ret_num_devices * sizeof(size_t) - 5);    
    clGetProgramInfo(f.program2D, CL_PROGRAM_BINARY_SIZES, f.ret_num_devices * sizeof(size_t), binary_sizes, NULL);
    ptx_code = (char**) malloc(f.ret_num_devices * sizeof(char*));
    for(i=0; i<f.ret_num_devices; ++i) {
        ptx_code[i]= (char*)malloc(binary_sizes[i]);
    }
    clGetProgramInfo(f.program2D, CL_PROGRAM_BINARIES, 0, ptx_code, NULL);
	fprintf(file1D, "%s\n", ptx_code[0]);
	#endif

	#if 1 /* Prepare 3D Program */
	char file3D[] = "src/kernels3D.cl";
	fp = fopen(file3D, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernels. Check path.\n");
		exit(1);
	}
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	f.program3D = clCreateProgramWithSource(	f.context, 1, 
												(const char **)&source_str, 
												(const size_t *)&source_size, 
												&f.error);

	// Build
	f.error = clBuildProgram(	f.program3D, 1, &f.device, 
								"-cl-nv-verbose", NULL, NULL);
	clGetProgramBuildInfo(	f.program3D, f.device, CL_PROGRAM_BUILD_LOG, 
							sizeof(buffer), buffer, NULL);
	clGetProgramBuildInfo(	f.program3D, f.device, CL_PROGRAM_BUILD_LOG, 0, 
							NULL, &ret_value_size);
	register1 = realloc(register1, ret_value_size+1);
	clGetProgramBuildInfo(	f.program3D, f.device, CL_PROGRAM_BUILD_LOG, 
							ret_value_size, register1, NULL);
	register1[ret_value_size] = '\0';
	fprintf(stderr, "%s\n", register1);

	file1D = fopen("src/kernels3D.ptx", "w");
    binary_sizes = (size_t*)malloc(f.ret_num_devices * sizeof(size_t) - 5);    
    clGetProgramInfo(f.program3D, CL_PROGRAM_BINARY_SIZES, f.ret_num_devices * sizeof(size_t), binary_sizes, NULL);
    ptx_code = (char**) malloc(f.ret_num_devices * sizeof(char*));
    for(i=0; i<f.ret_num_devices; ++i) {
        ptx_code[i]= (char*)malloc(binary_sizes[i]);
    }
    clGetProgramInfo(f.program3D, CL_PROGRAM_BINARIES, 0, ptx_code, NULL);
	fprintf(file1D, "%s\n", ptx_code[0]);
	#endif

	return 0;
}


