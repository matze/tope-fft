#include <CL/cl.h>
#include <assert.h>

#define MAX_SOURCE_SIZE (0x100000) // Eq: 1Mb
#define $CHECKERROR assert (f->error == CL_SUCCESS);	

#define C2C 1
#define R2C 0
#define FORWARD 1
#define INVERSE 0

struct topePlan1D {
	int x;			// Length
	int log;		
	int type;		// C2C, R2C, etc.
	int radix;
	cl_kernel kernel;		// butterfly kernel
	cl_kernel kernel_bit;	// bit reversal kernel
	cl_kernel kernel_swap;	// swapper kernel
	cl_kernel kernel_twid;	// twiddle calculator
	cl_kernel kernel_div;
	cl_mem data;			// main data
	cl_mem bitrev;			// bitreversal data
	cl_mem twiddle;			// twiddles
	size_t dataSize;		// size of data
	cl_ulong totalMemory;		// profiling: mem transfers
	cl_ulong totalPreKernel;	// profiling: before butterflies
	cl_ulong totalKernel;		// profiling: butterflies
	cl_uint dim;				// dimensions of data
	size_t *globalSize;			// kernel dimensions setup
	size_t *localSize;
};

struct topePlan2D {
	int x, y;					// dimensions
	int logX, logY;				// Log's of dimensions
	int type;					// C2C, R2C, etc.
	int radX, radY;				// Radices for each dimension;
	cl_kernel kernelX;			// Kernel for each dimension
	cl_kernel kernelY;			
	cl_kernel kernel_bit;		// Bit Reversals
	cl_kernel kernel_swap;		// Swapping
	cl_kernel kernel_twid;		// Twiddles
	cl_kernel kernel_div;		// Divide by Volume
	cl_mem data;				// main data
	cl_mem bitX, bitY;			// Bit reversal data
	cl_mem twdX, twdY;			// Twiddle Data
	size_t dataSize;			// Size of data
	cl_ulong totalMemory;		// profiling: mem transfers
	cl_ulong totalPreKernel;	// profiling: before butterflies
	cl_ulong totalKernel;		// profiling: butterflies
	cl_uint dim;				// dimensions of data
	size_t *globalSize;			// kernel dimensions setup
	size_t *localSize;		
};

struct topePlan3D {
	int x, y, z;				// dimensions
	int logX, logY, logZ;		// Log's of dimensions
	int type;					// C2C, R2C, etc.
	int radX, radY, radZ;		// Radices for each dimension;
	cl_kernel kernelX;			// Kernel for each dimension
	cl_kernel kernelY;			
	cl_kernel kernelZ;
	cl_kernel kernel_bit;		// Bit Reversals
	cl_kernel kernel_swap;		// Swapping
	cl_kernel kernel_twid;		// Twiddles
	cl_kernel kernel_div;		// Divide by Volume
	cl_mem data;				// main data
	cl_mem bitX, bitY, bitZ;	// Bit reversal data
	cl_mem twdX, twdY, twdZ;	// Twiddle Data
	size_t dataSize;			// Size of data
	cl_ulong totalMemory;		// profiling: mem transfers
	cl_ulong totalPreKernel;	// profiling: before butterflies
	cl_ulong totalKernel;		// profiling: butterflies
	cl_uint dim;				// dimensions of data
	size_t *globalSize;			// kernel dimensions setup
	size_t *localSize;		
};

struct topeFFT {
	cl_int 				error;
	cl_event			event;
	cl_platform_id 		platform_id;
	cl_device_id 		device;
	cl_program 			program1D;
	cl_program			program2D;
	cl_program			program3D;
	cl_context 			context;
	cl_command_queue 	command_queue;
	cl_uint 			ret_num_platforms;
	cl_uint 			ret_num_devices;
};

