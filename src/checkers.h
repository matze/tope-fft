#ifndef TOPE_CHECKERS_H
#define TOPE_CHECKERS_H

#include <CL/cl.h>

void clEnqueueNDRangeChecker(cl_int *err);
void clReleaseMemObjectChecker(cl_int *err);
void clEnqueueTaskChecker(cl_int *err);
void clGetPlatformIDsChecker(cl_int *err);
void clGetDeviceIDsChecker(cl_int *err);
void clCreateContextChecker(cl_int *err);
void clCreateCommandQueueChecker(cl_int *err);
void clCreateProgramWithSourceChecker(cl_int *err);
void clBuildProgramChecker(cl_int *err, char *p);
void clCreateKernelChecker(cl_int *err);
void clCreateBufferChecker(cl_int *err);
void clSetKernelArgChecker(cl_int *err);
void clEnqueueWriteBufferChecker(cl_int *err);
void clEnqueueReadBufferChecker(cl_int *err);
void clGetProgramBuildInfoChecker(cl_int *err);
void clEnqueueNDRangeChecker(cl_int *err);
void clReleaseMemObjectChecker(cl_int *err);

#endif
