#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void twid1D(__global double *twiddle, int size)
{
	int idX = get_global_id(0);
	double CLPI = acos(-1.);
	twiddle[2*idX] 	=  cos(2.*CLPI*idX/size);
	twiddle[2*idX+1] = -sin(2.*CLPI*idX/size);
}

__kernel void DIT4C2C(	__global double *data, 
						__global double *twiddle,
						const int size, unsigned int stage ) 
{

	int idX = get_global_id(0);
	
	double CLPI = acos(-1.);

	int powMaxLvl = 11;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int x;
	for (x = 0; x < powLevels; x++) {
		powX *= pow(4.0f,powMaxLvl);
	}
	powX *= pow(4.0f,powRemain);
	powXm1 = powX/4;

	int clipOne, clipTwo, clipThr, clipFou;
	int yIndex, kIndex;
	yIndex = idX / powXm1;
	kIndex = idX % powXm1;
	
	clipOne 	= 2 * (kIndex + yIndex * powX + 0 * powXm1);
	clipTwo 	= 2 * (kIndex + yIndex * powX + 1 * powXm1);
	clipThr		= 2 * (kIndex + yIndex * powX + 2 * powXm1);
	clipFou		= 2 * (kIndex + yIndex * powX + 3 * powXm1);

	double2 TEMPC;
	double8 dataLOC = (double8)(	data[clipOne+0],data[clipOne+1],
									data[clipTwo+0],data[clipTwo+1],
									data[clipThr+0],data[clipThr+1],
									data[clipFou+0],data[clipFou+1]	);

	int coeffUse = kIndex * size / powX;	
	int red = size/4;
	double2 clSet1;
	int quad = coeffUse/red;
	int buad = coeffUse%red;
	if (quad == 0) {
		clSet1.x = twiddle[2*coeffUse];
		clSet1.y = twiddle[2*coeffUse+1];
	}
	else if (quad == 1) {
		clSet1.x = twiddle[2*buad+1];
		clSet1.y = -twiddle[2*buad];
	}
	else if (quad == 2) {
		clSet1.x = -twiddle[2*buad];
		clSet1.y = -twiddle[2*buad+1];
	}
	double2 clSet2;
	quad = (2*coeffUse)/red;
	buad = (2*coeffUse)%red;
	if (quad == 0) {
		clSet2.x = twiddle[2*2*coeffUse];
		clSet2.y = twiddle[2*2*coeffUse+1];
	}
	else if (quad == 1) {
		clSet2.x = twiddle[2*buad+1];
		clSet2.y = -twiddle[2*buad];
	}
	else if (quad == 2) {
		clSet2.x = -twiddle[2*buad];
		clSet2.y = -twiddle[2*buad+1];
	}
	double2 clSet3;
	quad = (3*coeffUse)/red;
	buad = (3*coeffUse)%red;
	if (quad == 0) {
		clSet3.x = twiddle[2*3*coeffUse];
		clSet3.y = twiddle[2*3*coeffUse+1];
	}
	else if (quad == 1) {
		clSet3.x = twiddle[2*buad+1];
		clSet3.y = -twiddle[2*buad];
	}
	else if (quad == 2) {
		clSet3.x = -twiddle[2*buad];
		clSet3.y = -twiddle[2*buad+1];
	}

	#if 1
		TEMPC.x = dataLOC.s2 * clSet2.x - dataLOC.s3 * clSet2.y;
		TEMPC.y = dataLOC.s3 * clSet2.x + dataLOC.s2 * clSet2.y;
		dataLOC.s2 = TEMPC.x;
		dataLOC.s3 = TEMPC.y;
		
		TEMPC.x = dataLOC.s4 * clSet1.x - dataLOC.s5 * clSet1.y;
		TEMPC.y = dataLOC.s5 * clSet1.x + dataLOC.s4 * clSet1.y;
		dataLOC.s4 = TEMPC.x;
		dataLOC.s5 = TEMPC.y;

		TEMPC.x = dataLOC.s6 * clSet3.x - dataLOC.s7 * clSet3.y;
		TEMPC.y = dataLOC.s7 * clSet3.x + dataLOC.s6 * clSet3.y;
		dataLOC.s6 = TEMPC.x;
		dataLOC.s7 = TEMPC.y;
	#endif

	#if 0
	if (kIndex != 0) {
		clSet2.x = cos(two*two*CLPI*kIndex/powX);
		clSet2.y = mone*sin(two*two*CLPI*kIndex/powX);
		TEMPC.x = dataLOC.s2 * clSet2.x - dataLOC.s3 * clSet2.y;
		TEMPC.y = dataLOC.s3 * clSet2.x + dataLOC.s2 * clSet2.y;
		dataLOC.s2 = TEMPC.x;
		dataLOC.s3 = TEMPC.y;
		clSet1.x = cos(two*CLPI*kIndex/powX);
		clSet1.y = mone*sin(two*CLPI*kIndex/powX);
		TEMPC.x = dataLOC.s4 * clSet1.x - dataLOC.s5 * clSet1.y;
		TEMPC.y = dataLOC.s5 * clSet1.x + dataLOC.s4 * clSet1.y;
		dataLOC.s4 = TEMPC.x;
		dataLOC.s5 = TEMPC.y;
		clSet3.x = cos(3.0f*two*CLPI*kIndex/powX);
		clSet3.y = mone*sin(3.0f*two*CLPI*kIndex/powX);
		TEMPC.x = dataLOC.s6 * clSet3.x - dataLOC.s7 * clSet3.y;
		TEMPC.y = dataLOC.s7 * clSet3.x + dataLOC.s6 * clSet3.y;
		dataLOC.s6 = TEMPC.x;
		dataLOC.s7 = TEMPC.y;
	}	
	#endif

	data[clipOne+0] = dataLOC.s0 + dataLOC.s2 + dataLOC.s4 + dataLOC.s6;
	data[clipOne+1] = dataLOC.s1 + dataLOC.s3 + dataLOC.s5 + dataLOC.s7;
	data[clipTwo+0] = dataLOC.s0 - dataLOC.s2 + dataLOC.s5 - dataLOC.s7;
	data[clipTwo+1] = dataLOC.s1 - dataLOC.s3 - dataLOC.s4 + dataLOC.s6;
	data[clipThr+0] = dataLOC.s0 + dataLOC.s2 - dataLOC.s4 - dataLOC.s6;
	data[clipThr+1] = dataLOC.s1 + dataLOC.s3 - dataLOC.s5 - dataLOC.s7;
	data[clipFou+0] = dataLOC.s0 - dataLOC.s2 - dataLOC.s5 + dataLOC.s7;
	data[clipFou+1] = dataLOC.s1 - dataLOC.s3 + dataLOC.s4 - dataLOC.s6;
	#if 0
	data[clipOne+0] = 0;//
	data[clipOne+1] = 0;
	data[clipTwo+0] = clSet2.x;
	data[clipTwo+1] = clSet2.y;
	data[clipThr+0] = clSet1.x;
	data[clipThr+1] = clSet1.y;
	data[clipFou+0] = clSet3.x;
	data[clipFou+1] = clSet3.y;
	#endif




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

