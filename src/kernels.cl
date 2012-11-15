#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void twid1D(__global double *twiddle, int size)
{
	int idX = get_global_id(0);
	double CLPI = acos(-1.);
	twiddle[2*idX] 	=  cos(2.*CLPI*idX/size);
	twiddle[2*idX+1] = -sin(2.*CLPI*idX/size);
}

__kernel void DIT2C2C(	__global double *data, 
							__global double *twiddle,
							const int size, unsigned int stage ) 
{
	#if 1
	int idX = get_global_id(0);

	int powMaxLvl = 11;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int x;
	for (x = 0; x < powLevels; x++)	powX *= pow(2.0f,powMaxLvl);
	powX *= pow(2.0f,powRemain);
	powXm1 = powX/2;

	double CLPI = acos(-1.);

	int yIndex = idX / powXm1;
	int kIndex = idX % powXm1;

	int clipStart 	= 2*(kIndex + yIndex * powX);
	int clipEnd 	= 2*(kIndex + yIndex * powX + powXm1);
	int coeffUse 	= kIndex * (size/powX);

	int red = size/4;
	int quad = coeffUse/red;
	int buad = coeffUse%red;
	double2 CLCOSSIN;
	if (quad == 0) {
		CLCOSSIN.x = twiddle[2*coeffUse];
		CLCOSSIN.y = twiddle[2*coeffUse+1];
	}
	else if (quad == 1) {
		CLCOSSIN.x = twiddle[2*buad+1];
		CLCOSSIN.y = -twiddle[2*buad];
	}

	double4 LOC = (double4)(	data[clipStart],data[clipStart+1],
								data[clipEnd],	data[clipEnd+1]);
	double4 FIN = (double4)(	LOC.x + LOC.z * CLCOSSIN.x - LOC.w * CLCOSSIN.y,
								LOC.y + LOC.w * CLCOSSIN.x + LOC.z * CLCOSSIN.y,
								LOC.x - LOC.z * CLCOSSIN.x + LOC.w * CLCOSSIN.y,
								LOC.y - LOC.w * CLCOSSIN.x - LOC.z * CLCOSSIN.y);

	#if 1
	data[clipStart] 	= FIN.x;
	data[clipStart+1] 	= FIN.y;
	data[clipEnd] 		= FIN.z;
	data[clipEnd+1] 	= FIN.w;
	#endif
	#if 0
	if (quad == 0) {
		[2*idX] = cos(two*CLPI*(coeffUse)/xR);
		[2*idX+1] = -sin(two*CLPI*(coeffUse)/xR);
		[2*xR/2+2*idX] = twiddle[2*coeffUse];
		[2*xR/2+2*idX+1] = twiddle[2*coeffUse+1];
	}
	else if (quad == 1) {
		[2*idX] = cos(two*CLPI*(coeffUse)/xR);
		[2*idX+1] = -sin(two*CLPI*(coeffUse)/xR);
		[2*xR/2+2*idX] = twiddle[2*buad+1];
		[2*xR/2+2*idX+1] = -twiddle[2*buad];
	}
	#endif
	#endif
}

__kernel void swap1D(	__global double *data, 
						__global int *bitRev) 
{
	int idX = get_global_id(0);
	double holder;
	int old = 0, new = 0;

	if (idX < bitRev[idX]) {
		old = 2*idX;
		new = 2*bitRev[idX];

		holder = data[new];
		data[new] = data[old];
		data[old] = holder;

		holder = data[new+1];
		data[new+1] = data[old+1];
		data[old+1] = holder;
	}
}

__kernel void reverse2(__global int *bitRev, int logSize)
{
	int global_id = get_global_id(0);

	int powMaxLvl = 11;
	int powLevels, powRemain, powX, x;

	int i, j, andmask, sum = 0, k;
	for (i = logSize - 1, j = 0; i >= 0; i--, j++) {
		andmask = 1 << i;
		k = global_id & andmask;

		powLevels = j / powMaxLvl;
		powRemain = j % powMaxLvl;
		powX = 1;
		for (x = 0; x < powLevels; x++) 
			powX *= pow(2.0f,powMaxLvl);
		powX *= pow(2.0f,powRemain);
		sum += k == 0 ? 0 : powX;
	}
	bitRev[global_id] = sum;
}

