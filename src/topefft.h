#include <CL/cl.h>
#include <assert.h>

#define MAX_SOURCE_SIZE (0x100000) // Eq: 1Mb
#define $CHECKERROR assert (f->error == CL_SUCCESS);	

#define C2C 1
#define R2C 0
#define FORWARD 1
#define INVERSE 0

struct topePlan1D {
	int x;
	int log;
	int type;
	int radix;
	cl_kernel kernel;
	cl_kernel kernel_bit;
	cl_kernel kernel_swap;
	cl_kernel kernel_twid;
	cl_mem data;
	cl_mem bitrev;
	cl_mem twiddle;
	size_t dataSize;
	cl_ulong totalMemory;
	cl_ulong totalPreKernel;
	cl_ulong totalKernel;
	cl_uint dim;
	size_t *globalSize;
	size_t *localSize;
};

struct topeFFT {
	cl_int 				error;
	cl_event			event;
	cl_platform_id 		platform_id;
	cl_device_id 		device;
	cl_context 			context;
	cl_command_queue 	command_queue;
	cl_program 			program;
	cl_uint 			ret_num_platforms;
	cl_uint 			ret_num_devices;
};

