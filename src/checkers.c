#include <CL/cl.h>	// OpenCL Specification
#include <stdio.h>
#include <stdlib.h>

#define SUCCESS 0

void clEnqueueNDRangeChecker(cl_int *err) {
	if (*err == CL_INVALID_PROGRAM_EXECUTABLE) fprintf(stderr,"Err (clEnqueueNDRange): No successfully built program executable available\n");
	else if (*err == CL_INVALID_COMMAND_QUEUE) fprintf(stderr,"Err (clEnqueueNDRange): Okay so the command queue is not right\n");
	else if (*err == CL_INVALID_KERNEL) fprintf(stderr,"Err (clEnqueueNDRange): Invalid Kernel Object\n");
	else if (*err == CL_INVALID_CONTEXT) fprintf(stderr,"Err (clEnqueueNDRange): Invalid Context\n");	// New
	else if (*err == CL_INVALID_KERNEL_ARGS)	fprintf(stderr,"Err (clEnqueueNDRange): Invalid Kernel Arguments\n");
	else if (*err == CL_INVALID_WORK_DIMENSION) fprintf(stderr,"Err (clEnqueueNDRange): IF WORK DIMENSIONS IS NOT VALID VALUE THAT IS BETWEEN 1 AND 3\n");
	else if (*err == CL_INVALID_GLOBAL_WORK_SIZE) fprintf(stderr,"Err (clEnqueueNDRange): Invalid Global Work Size. Check for 0 or NULL\n"); // New
	else if (*err == CL_INVALID_GLOBAL_OFFSET) fprintf(stderr,"Err (clEnqueueNDRange): Offset greater or invalid\n"); // New
	else if (*err == CL_INVALID_WORK_GROUP_SIZE) {
		fprintf(stderr,"Err (clEnqueueNDRange): IF WORK GROUP SIZE IS NOT EVENLY DIVISIABLE BY LOCAL WORK SIZE\n");
		exit(0);
	}
	else if (*err == CL_INVALID_WORK_ITEM_SIZE) fprintf(stderr,"Err (clEnqueueNDRange): IF NUMBER OF ITEMS IS GREATER THAN CL_DEVICE_MAX_WORK_ITEM_SIZES\n");
	else if (*err == CL_MISALIGNED_SUB_BUFFER_OFFSET) fprintf(stderr,"Err (clEnqueueNDRange): Sub Buffer Offset error or misaligned\n"); // New
	else if (*err == CL_INVALID_IMAGE_SIZE) fprintf(stderr,"Err (clEnqueueNDRange): Invalid Image Size\n"); // New
	else if (*err == CL_OUT_OF_RESOURCES) fprintf(stderr,"Err (clEnqueueNDRange): Insufficient resources\n");
	else if (*err == CL_MEM_OBJECT_ALLOCATION_FAILURE) fprintf(stderr,"Err (clEnqueueNDRange): Failure to allocate data storage mem associated with image or buffer objects\n");
	else if (*err == CL_INVALID_EVENT_WAIT_LIST) fprintf(stderr,"Err (clEnqueueNDRange): Event list is NULL but number in event list is > 0\n"); // New
	else if (*err == CL_OUT_OF_HOST_MEMORY) fprintf(stderr,"Err (clEnqueueNDRange): Failure to allocate resources on host\n"); // New
	#if SUCCESS
	else if (*err == CL_SUCCESS) fprintf(stderr,"... Executing Kernel\n");
	#endif
	else {
		if (*err != CL_SUCCESS) fprintf(stderr,"Err (clEnqueueNDRange): Unknown Error\n");
	}	
}

void clReleaseMemObjectChecker(cl_int *err) {
	if (*err == CL_INVALID_MEM_OBJECT) fprintf(stderr, "Err (clReleaseMemObject): Not Valid Mem Object\n");
	#if SUCCESS
	else if (*err == CL_SUCCESS) fprintf(stderr,"... Mem Object Deleted\n");
	#endif
	else {
		if (*err != CL_SUCCESS) fprintf(stderr, "Err (clReleaseMemObject): Failure in deleting mem object\n");
	}
}

void clEnqueueTaskChecker(cl_int *err) {
	if (*err == CL_INVALID_PROGRAM_EXECUTABLE) fprintf(stderr,"Err (clEnqueueTask): No successfully built program executable available\n");
	else if (*err == CL_INVALID_COMMAND_QUEUE) fprintf(stderr,"Err (clEnqueueTask): Okay so the command queue is not right\n");
	else if (*err == CL_INVALID_KERNEL) fprintf(stderr,"Err (clEnqueueTask): Invalid Kernel Object\n");
	else if (*err == CL_INVALID_KERNEL_ARGS) fprintf(stderr,"Err (clEnqueueTask): Invalid Kernel Arguments\n");
	else if (*err == CL_OUT_OF_RESOURCES) fprintf(stderr,"Err (clEnqueueTask): Insufficient resources\n");
	else if (*err == CL_MEM_OBJECT_ALLOCATION_FAILURE) fprintf(stderr,"Err (clEnqueueTask): Failure to allocate memory for data storage associated with image or buffer objects\n");
	#if SUCCESS
	else if (*err == CL_SUCCESS) fprintf(stderr,"... EnqueueTaskChecker Success\n");
	#endif
	else {
		if (*err != CL_SUCCESS) fprintf(stderr,"Err (clEnqueueTask): Unknown Error\n");
	}
}

void clGetPlatformIDsChecker(cl_int *err) {
	if (*err != CL_SUCCESS) fprintf(stderr,"clGetPlatformIDsChecker\n");
	#if SUCCESS
  	else if (*err == CL_SUCCESS) fprintf(stderr,"... Obtained Platforms\n");
	#endif
}

void clGetDeviceIDsChecker(cl_int *err) {
	if (*err == CL_INVALID_PLATFORM) fprintf(stderr,"Err (clGetDeviceIDs): Platform not valid\n");
	else if (*err == CL_INVALID_DEVICE_TYPE) fprintf(stderr,"Err (clGetDeviceIDs): Device type not valie\n");
	else if (*err == CL_INVALID_VALUE)	fprintf(stderr,"Err (clGetDeviceIDs): 3rd arg 0, but devices != NULL, or no devices found\n");
	else if (*err == CL_DEVICE_NOT_FOUND) fprintf(stderr,"Err (clGetDeviceIDs): No device matching arg2 found\n");
	#if SUCCESS
	else if (*err == CL_SUCCESS) fprintf(stderr,"... Device IDs Checker Success\n");
	#endif
	else {
		if (*err != CL_SUCCESS) fprintf(stderr,"clGetDeviceIDsChecker\n");
	}
}

void clCreateContextChecker(cl_int *err) {
	if (*err == CL_INVALID_PLATFORM) fprintf(stderr,"Err (clCreateContext): Properties Null | No Platform can be selected | Not valid platform\n");
	else if (*err == CL_INVALID_VALUE) fprintf(stderr,"Err (clCreateContext): Devices = NULL | number of devices = 0 | Invalid Device\n");
	else if (*err == CL_DEVICE_NOT_AVAILABLE) fprintf(stderr,"Err (clCreateContext): Device Busy. Wait and try again\n");
	else if (*err == CL_OUT_OF_HOST_MEMORY) fprintf(stderr,"Err (clCreateContext): Error allocating resources\n");
	#if SUCCESS
	else if (*err == CL_SUCCESS) fprintf(stderr,"... Context Checker Success\n");
	#endif
	else {
		if (*err != CL_SUCCESS) fprintf(stderr,"clCreateContextChecker\n");
	}
}

void clCreateCommandQueueChecker(cl_int *err) {
	if (*err == CL_INVALID_CONTEXT) fprintf(stderr,"Err (clCreateCommandQueue): Invalid Context\n");
	else if (*err == CL_INVALID_DEVICE) fprintf(stderr,"Err (clCreateCommandQueue): Invalid Device, or Device not associated with context\n");
	else if (*err == CL_INVALID_VALUE) fprintf(stderr,"Err (clCreateCommandQueue): Properties (arg3) must be either out of order, or in order, or with/without profiling support\n");
	else if (*err == CL_OUT_OF_HOST_MEMORY) fprintf(stderr,"Err (clCreateCommandQueue): Error allocating resources\n");
	#if SUCCESS
	if (*err == CL_SUCCESS) fprintf(stderr,"... CommandQueue Created\n");
	#endif
	else {
		if (*err != CL_SUCCESS) fprintf(stderr,"clCreateCommandQueueChecker\n");
	}
}

void clCreateProgramWithSourceChecker(cl_int *err) {
	if (*err == CL_INVALID_CONTEXT) fprintf(stderr,"Err (clCreateProgramWithSource): Context is not valid context\n");
	else if (*err == CL_INVALID_VALUE) fprintf(stderr,"Err (clCreateProgramWithSource): Count (arg2) is 0 | Contents of source code is NULL\n");
	else if (*err == CL_OUT_OF_HOST_MEMORY) fprintf(stderr,"Err (clCreateProgramWithSource): Resource (Memory) Allocation Failure on Host\n");
	#if SUCCESS
	else if (*err == CL_SUCCESS) fprintf(stderr,"... Created Program from Source\n");
	#endif
	else {
		if (*err != CL_SUCCESS) fprintf(stderr,"clCreateProgramWithSourceChecker\n");
	}
}

void clBuildProgramChecker(cl_int *err, char *p) {
	if (*err != CL_SUCCESS) {
		if (*err == CL_INVALID_PROGRAM) fprintf(stderr,"Err (clBuildProgram): Program is not valid Program Object\n");
		else if (*err == CL_INVALID_VALUE) fprintf(stderr,"Err (clBuildProgram): Device list is NULL and number of devices >0 ... Incompatible\n");
		else if (*err == CL_INVALID_DEVICE) fprintf(stderr,"Err (clBuildProgram): Device specified does not match the device specified in program object\n");
		else if (*err == CL_INVALID_BUILD_OPTIONS) fprintf(stderr,"Err (clBuildProgram): Invalid Build Options\n");
		else if (*err == CL_BUILD_PROGRAM_FAILURE) fprintf(stderr,"Err (clBuildProgram): Failure in building the program\n");
		else fprintf(stderr,"clBuildProgramChecker\n");
		fprintf(stderr,"Build Log: %s\n", p);
		exit(0);
	}
	#if SUCCESS
	else if (*err == CL_SUCCESS) fprintf(stderr,"... Build Success\n");
	#endif
}

void clCreateKernelChecker(cl_int *err) {
	if (*err == CL_INVALID_PROGRAM) fprintf(stderr,"Err (clCreateKernel): Program is not valid Program Object\n");
	else if (*err == CL_INVALID_PROGRAM_EXECUTABLE) fprintf(stderr,"Err (clCreateKernel): Kernel Binary is Invalid\n");
	else if (*err == CL_INVALID_KERNEL_NAME) fprintf(stderr,"Err (clCreateKernel): Invalid Kernel Name in Program Object\n");
	else if (*err == CL_INVALID_VALUE) fprintf(stderr,"Err (clCreateKernel): Kernel Name is NULL\n");
	else if (*err == CL_INVALID_KERNEL_DEFINITION) fprintf(stderr,"Err (clCreateKernel): Function Definition defined for Kernel is not conformant with device\n");
	#if SUCCESS
	else if (*err == CL_SUCCESS) fprintf(stderr,"... Created Kernel\n");
	#endif
	else {
		if (*err != CL_SUCCESS) fprintf(stderr,"clCreateKernelChecker\n");
	}
}

void clCreateBufferChecker(cl_int *err) {
	if (*err == CL_INVALID_CONTEXT) fprintf(stderr,"Err (clCreateBuffer): Context is not valid context\n");
	else if (*err == CL_INVALID_VALUE) fprintf(stderr,"Err (clCreateBuffer): Flag (arg2) is not valid\n");
	else if (*err == CL_INVALID_BUFFER_SIZE) fprintf(stderr,"Err (clCreateBuffer): Size is 0 !! or is greater than maximum allocatable size\n");
	else if (*err == CL_MEM_OBJECT_ALLOCATION_FAILURE) fprintf(stderr,"Err (clCreateBuffer): Failure to allocate memory for buffer object\n");
	#if SUCCESS
	else if (*err == CL_SUCCESS) fprintf(stderr,"... Created Buffer\n");
	#endif
	else {
		if (*err != CL_SUCCESS) fprintf(stderr,"clCreateBufferChecker\n");
	}
}

void clSetKernelArgChecker(cl_int *err) {
	if (*err == CL_INVALID_KERNEL) fprintf(stderr,"Err (clSetKernelArg): Kernel is not valid Kernel object\n");
	else if (*err == CL_INVALID_ARG_INDEX) fprintf(stderr,"Err (clSetKernelArg): You sure arg2 refering to the right input parameter?\n");
	else if (*err == CL_INVALID_MEM_OBJECT) fprintf(stderr,"Err (clSetKernelArg): Invalid Memory Object associated with Kernel Argument\n");
	else if (*err == CL_INVALID_ARG_VALUE) fprintf(stderr,"Err (clSetKernelArg): Invalid Argument Value. Check API for details\n");
	else if (*err == CL_INVALID_SAMPLER) fprintf(stderr,"Err (clSetKernelArg): If argument is sampler_t declared but actually it isn't\n");
	else if (*err == CL_INVALID_ARG_SIZE) fprintf(stderr,"Err (clSetKernelArg): If arg_size does not match the size of the data for argument\n");
	#if SUCCESS
	else if (*err == CL_SUCCESS) fprintf(stderr,"... Kernel Arg set\n");
	#endif
	else {
		if (*err != CL_SUCCESS) fprintf(stderr,"clSetKernelArgChecker\n");
	}
}

void clEnqueueWriteBufferChecker(cl_int *err) {
	if (*err == CL_INVALID_MEM_OBJECT) fprintf(stderr, "Err (clEnqueueWriteBuffer): buffer is not valid buffer object\n");
	else if (*err == CL_INVALID_VALUE) fprintf(stderr, "Err (clEnqueueWriteBuffer): region is out of bounds or ptr is NULL\n");
	else if (*err == CL_INVALID_EVENT_WAIT_LIST) fprintf(stderr, "Err (clEnqueueWriteBuffer): Events are not valid events\n");
	else if (*err == CL_OUT_OF_HOST_MEMORY) fprintf(stderr, "Err (clEnqueueWriteBuffer): Failure to allocate resources at runtime\n");
	else if (*err == CL_INVALID_COMMAND_QUEUE) fprintf(stderr, "Err (clEnqueueWriteBuffer): Command Queue not valid\n");
	else if (*err == CL_INVALID_CONTEXT) fprintf(stderr, "Err (clEnqueueWriteBuffer): Context is not correct. Check it\n");
	else if (*err == CL_MEM_OBJECT_ALLOCATION_FAILURE) fprintf(stderr, "Err (clEnqueueWriteBuffer): Failure to allocate memory for buffer\n");
	#if SUCCESS
	else if (*err == CL_SUCCESS) fprintf(stderr,"... Write Buffer Success\n");
	#endif
	else {
		if (*err != CL_SUCCESS) fprintf(stderr,"clEnqueueWriteBufferChecker\n");
	}
}

void clEnqueueReadBufferChecker(cl_int *err) {
	if (*err == CL_INVALID_COMMAND_QUEUE) fprintf(stderr,"Err (EnqueueReadBuffer): The command queue is invalid\n");
	else if (*err == CL_INVALID_CONTEXT) fprintf(stderr,"Err (EnqueueReadBuffer): The context associated with command queue and buffer are not the same\n");
	else if (*err == CL_INVALID_MEM_OBJECT) fprintf(stderr,"Err (EnqueueReadBuffer): Buffer is not a valid buffer object\n");
	else if (*err == CL_INVALID_VALUE) fprintf(stderr,"Err (EnqueueReadBuffer): The region being written is out of bounds or pointer is of NULL value\n");
	#if SUCCESS
	else if (*err == CL_SUCCESS) fprintf(stderr,"... Read Buffer Success\n");
	#endif
	else if (*err == CL_INVALID_EVENT_WAIT_LIST) fprintf(stderr,"Err (EnqueueReadBuffer): Something wrong with theh wait list\n");
	else if (*err == CL_MEM_OBJECT_ALLOCATION_FAILURE) fprintf(stderr,"Err (EnqueueReadBuffer): Failure to alllocate memory buffer\n");
	else if (*err == CL_OUT_OF_HOST_MEMORY) fprintf(stderr,"Err (EnqueueReadBuffer): Failure to allocate resources\n");
	else {
		if (*err != CL_SUCCESS) fprintf(stderr,"clEnqueueReadBufferChecker (%d, %d)\n", (int)*err, (int)CL_OUT_OF_RESOURCES);
	}
}

void clGetProgramBuildInfoChecker(cl_int *err) {
	if (*err == CL_INVALID_DEVICE) fprintf(stderr,"Err (clGetProgramBuildInfo): Device is not in list of devices associated with program\n");
	else if (*err == CL_INVALID_VALUE) fprintf(stderr,"Err (clGetProgramBuildInfo): param_name is not valid or param_value_size is less than return type\n");
	else if (*err == CL_INVALID_PROGRAM) fprintf(stderr,"Err (clGetProgramBuildInfo): program is not a valid program object\n");
	#if SUCCESS
	else if (*err == CL_SUCCESS) fprintf(stderr, "... Build Info Success\n");
	#endif
	else {
		if (*err != CL_SUCCESS) fprintf(stderr,"clGetProgramBuildInfoChecker\n");
	}
}


