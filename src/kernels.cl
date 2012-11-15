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
	double8 SIGA = (double8)(	data[clipOne+0],data[clipOne+1],
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
		TEMPC.x = SIGA.s2 * clSet2.x - SIGA.s3 * clSet2.y;
		TEMPC.y = SIGA.s3 * clSet2.x + SIGA.s2 * clSet2.y;
		SIGA.s2 = TEMPC.x;
		SIGA.s3 = TEMPC.y;
		
		TEMPC.x = SIGA.s4 * clSet1.x - SIGA.s5 * clSet1.y;
		TEMPC.y = SIGA.s5 * clSet1.x + SIGA.s4 * clSet1.y;
		SIGA.s4 = TEMPC.x;
		SIGA.s5 = TEMPC.y;

		TEMPC.x = SIGA.s6 * clSet3.x - SIGA.s7 * clSet3.y;
		TEMPC.y = SIGA.s7 * clSet3.x + SIGA.s6 * clSet3.y;
		SIGA.s6 = TEMPC.x;
		SIGA.s7 = TEMPC.y;
	#endif

	#if 0
	if (kIndex != 0) {
		clSet2.x = cos(two*two*CLPI*kIndex/powX);
		clSet2.y = mone*sin(two*two*CLPI*kIndex/powX);
		TEMPC.x = SIGA.s2 * clSet2.x - SIGA.s3 * clSet2.y;
		TEMPC.y = SIGA.s3 * clSet2.x + SIGA.s2 * clSet2.y;
		SIGA.s2 = TEMPC.x;
		SIGA.s3 = TEMPC.y;
		clSet1.x = cos(two*CLPI*kIndex/powX);
		clSet1.y = mone*sin(two*CLPI*kIndex/powX);
		TEMPC.x = SIGA.s4 * clSet1.x - SIGA.s5 * clSet1.y;
		TEMPC.y = SIGA.s5 * clSet1.x + SIGA.s4 * clSet1.y;
		SIGA.s4 = TEMPC.x;
		SIGA.s5 = TEMPC.y;
		clSet3.x = cos(3.0f*two*CLPI*kIndex/powX);
		clSet3.y = mone*sin(3.0f*two*CLPI*kIndex/powX);
		TEMPC.x = SIGA.s6 * clSet3.x - SIGA.s7 * clSet3.y;
		TEMPC.y = SIGA.s7 * clSet3.x + SIGA.s6 * clSet3.y;
		SIGA.s6 = TEMPC.x;
		SIGA.s7 = TEMPC.y;
	}	
	#endif

	data[clipOne+0] = SIGA.s0 + SIGA.s2 + SIGA.s4 + SIGA.s6;
	data[clipOne+1] = SIGA.s1 + SIGA.s3 + SIGA.s5 + SIGA.s7;
	data[clipTwo+0] = SIGA.s0 - SIGA.s2 + SIGA.s5 - SIGA.s7;
	data[clipTwo+1] = SIGA.s1 - SIGA.s3 - SIGA.s4 + SIGA.s6;
	data[clipThr+0] = SIGA.s0 + SIGA.s2 - SIGA.s4 - SIGA.s6;
	data[clipThr+1] = SIGA.s1 + SIGA.s3 - SIGA.s5 - SIGA.s7;
	data[clipFou+0] = SIGA.s0 - SIGA.s2 - SIGA.s5 + SIGA.s7;
	data[clipFou+1] = SIGA.s1 - SIGA.s3 + SIGA.s4 - SIGA.s6;
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



__kernel void DIT8C2C(	__global double *data, 
						__global double *twiddle,
						const int size, unsigned int stage ) 
{
	int idX = get_global_id(0);
	double two = 2.0;
	double mone = 0 - 1.0;
	double CLPI = acos(mone);

	int powMaxLvl = 4;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int x;
	for (x = 0; x < powLevels; x++) {
		powX *= pow(8.0f,powMaxLvl);
	}
	powX *= pow(8.0f,powRemain);
	powXm1 = powX/8;

	int clipOne, clipTwo, clipThr, clipFou, clipFiv, clipSix, clipSev, clipEig;
	int yIndex, kIndex;
	yIndex = idX / powXm1;
	kIndex = idX % powXm1;
	
	clipOne 	= 2 * (kIndex + yIndex * powX + 0 * powXm1);
	clipTwo 	= 2 * (kIndex + yIndex * powX + 1 * powXm1);
	clipThr		= 2 * (kIndex + yIndex * powX + 2 * powXm1);
	clipFou		= 2 * (kIndex + yIndex * powX + 3 * powXm1);
	clipFiv		= 2 * (kIndex + yIndex * powX + 4 * powXm1);
	clipSix		= 2 * (kIndex + yIndex * powX + 5 * powXm1);
	clipSev		= 2 * (kIndex + yIndex * powX + 6 * powXm1);
	clipEig		= 2 * (kIndex + yIndex * powX + 7 * powXm1);

	double2 CST;
	double2 TMP;
	double16 SIGA = (double16)(	data[clipOne+0],data[clipOne+1],	// s0, s1
								data[clipTwo+0],data[clipTwo+1],	// s2, s3
								data[clipThr+0],data[clipThr+1],	// s4, s5
								data[clipFou+0],data[clipFou+1],	// s6, s7
								data[clipFiv+0],data[clipFiv+1],	// s8, s9
								data[clipSix+0],data[clipSix+1],	// sa, sb
								data[clipSev+0],data[clipSev+1],	// sc, sd
								data[clipEig+0],data[clipEig+1]);	// se, sf

	
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
	double2 clSet4;
	quad = (4*coeffUse)/red;
	buad = (4*coeffUse)%red;
	if (quad == 0) {
		clSet4.x = twiddle[2*4*coeffUse];
		clSet4.y = twiddle[2*4*coeffUse+1];
	}
	else if (quad == 1) {
		clSet4.x = twiddle[2*buad+1];
		clSet4.y = -twiddle[2*buad];
	}
	else if (quad == 2) {
		clSet4.x = -twiddle[2*buad];
		clSet4.y = -twiddle[2*buad+1];
	}

	double2 clSet5;
	quad = (5*coeffUse)/red;
	buad = (5*coeffUse)%red;
	if (quad == 0) {
		clSet5.x = twiddle[2*5*coeffUse];
		clSet5.y = twiddle[2*5*coeffUse+1];
	}
	else if (quad == 1) {
		clSet5.x = twiddle[2*buad+1];
		clSet5.y = -twiddle[2*buad];
	}
	else if (quad == 2) {
		clSet5.x = -twiddle[2*buad];
		clSet5.y = -twiddle[2*buad+1];
	}

	double2 clSet6;
	quad = (6*coeffUse)/red;
	buad = (6*coeffUse)%red;
	if (quad == 0) {
		clSet6.x = twiddle[2*6*coeffUse];
		clSet6.y = twiddle[2*6*coeffUse+1];
	}
	else if (quad == 1) {
		clSet6.x = twiddle[2*buad+1];
		clSet6.y = -twiddle[2*buad];
	}
	else if (quad == 2) {
		clSet6.x = -twiddle[2*buad];
		clSet6.y = -twiddle[2*buad+1];
	}

	double2 clSet7;
	quad = (7*coeffUse)/red;
	buad = (7*coeffUse)%red;
	if (quad == 0) {
		clSet7.x = twiddle[2*7*coeffUse];
		clSet7.y = twiddle[2*7*coeffUse+1];
	}
	else if (quad == 1) {
		clSet7.x = twiddle[2*buad+1];
		clSet7.y = -twiddle[2*buad];
	}
	else if (quad == 2) {
		clSet7.x = -twiddle[2*buad];
		clSet7.y = -twiddle[2*buad+1];
	}
	else if (quad == 3) {
		clSet7.x = -twiddle[2*buad+1];
		clSet7.y =  twiddle[2*buad];
	}






	
	if (kIndex != 0) {
		TMP.x = SIGA.s2 * clSet4.x - SIGA.s3 * clSet4.y;
		TMP.y = SIGA.s2 * clSet4.y + SIGA.s3 * clSet4.x;
		SIGA.s2 = TMP.x;
		SIGA.s3 = TMP.y;

		TMP.x = SIGA.s4 * clSet2.x - SIGA.s5 * clSet2.y;
		TMP.y = SIGA.s4 * clSet2.y + SIGA.s5 * clSet2.x;
		SIGA.s4 = TMP.x;
		SIGA.s5 = TMP.y;

		TMP.x = SIGA.s6 * clSet6.x - SIGA.s7 * clSet6.y;
		TMP.y = SIGA.s6 * clSet6.y + SIGA.s7 * clSet6.x;
		SIGA.s6 = TMP.x;
		SIGA.s7 = TMP.y;

		TMP.x = SIGA.s8 * clSet1.x - SIGA.s9 * clSet1.y;
		TMP.y = SIGA.s8 * clSet1.y + SIGA.s9 * clSet1.x;
		SIGA.s8 = TMP.x;
		SIGA.s9 = TMP.y;

		TMP.x = SIGA.sa * clSet5.x - SIGA.sb * clSet5.y;
		TMP.y = SIGA.sa * clSet5.y + SIGA.sb * clSet5.x;
		SIGA.sa = TMP.x;
		SIGA.sb = TMP.y;

		TMP.x = SIGA.sc * clSet3.x - SIGA.sd * clSet3.y;
		TMP.y = SIGA.sc * clSet3.y + SIGA.sd * clSet3.x;
		SIGA.sc = TMP.x;
		SIGA.sd = TMP.y;
		
		TMP.x = SIGA.se * clSet7.x - SIGA.sf * clSet7.y;
		TMP.y = SIGA.se * clSet7.y + SIGA.sf * clSet7.x;
		SIGA.se = TMP.x;
		SIGA.sf = TMP.y;
	}	

	double tmp;
	double d707 = cos(CLPI/4.);

	double16 SIGB = (double16)(	SIGA.s0 + SIGA.s2,
								SIGA.s1 + SIGA.s3,
								SIGA.s0 - SIGA.s2,
								SIGA.s1 - SIGA.s3,
								SIGA.s4 + SIGA.s6,
								SIGA.s5 + SIGA.s7,
								SIGA.s4 - SIGA.s6,
								SIGA.s5 - SIGA.s7,
								SIGA.s8 + SIGA.sa,
								SIGA.s9 + SIGA.sb,
								SIGA.s8 - SIGA.sa,
								SIGA.s9 - SIGA.sb,
								SIGA.sc + SIGA.se,
								SIGA.sd + SIGA.sf,
								SIGA.sc - SIGA.se,
								SIGA.sd - SIGA.sf);

	tmp = (SIGB.sa + SIGB.sb) * d707;
	SIGB.sb = (SIGB.sb - SIGB.sa) * d707;
	SIGB.sa = tmp;
	tmp = (SIGB.sf - SIGB.se) * d707;
	SIGB.sf = (SIGB.sf + SIGB.se) * -d707;
	SIGB.se = tmp;
	tmp = SIGB.s7; SIGB.s7 = -SIGB.s6; SIGB.s6 = tmp;

	SIGA.s0 = SIGB.s0 + SIGB.s4;
	SIGA.s1 = SIGB.s1 + SIGB.s5;
	SIGA.s2 = SIGB.s2 + SIGB.s6;
	SIGA.s3 = SIGB.s3 + SIGB.s7;
	SIGA.s4 = SIGB.s0 - SIGB.s4;
	SIGA.s5 = SIGB.s1 - SIGB.s5;
	SIGA.s6 = SIGB.s2 - SIGB.s6;
	SIGA.s7 = SIGB.s3 - SIGB.s7;
	SIGA.s8 = SIGB.s8 + SIGB.sc;
	SIGA.s9 = SIGB.s9 + SIGB.sd;
	SIGA.sa = SIGB.sa + SIGB.se;
	SIGA.sb = SIGB.sb + SIGB.sf;
	SIGA.sc = SIGB.s9 - SIGB.sd;
	SIGA.sd = SIGB.sc - SIGB.s8;
	SIGA.se = SIGB.sb - SIGB.sf;
	SIGA.sf = SIGB.se - SIGB.sa;

	#if 1
	data[clipOne+0] = SIGA.s0 + SIGA.s8;
	data[clipOne+1] = SIGA.s1 + SIGA.s9;
	data[clipTwo+0] = SIGA.s2 + SIGA.sa;
	data[clipTwo+1] = SIGA.s3 + SIGA.sb;
	data[clipThr+0] = SIGA.s4 + SIGA.sc;
	data[clipThr+1] = SIGA.s5 + SIGA.sd;
	data[clipFou+0] = SIGA.s6 + SIGA.se;
	data[clipFou+1] = SIGA.s7 + SIGA.sf;
	data[clipFiv+0] = SIGA.s0 - SIGA.s8;
	data[clipFiv+1] = SIGA.s1 - SIGA.s9;
	data[clipSix+0] = SIGA.s2 - SIGA.sa;
	data[clipSix+1] = SIGA.s3 - SIGA.sb;
	data[clipSev+0] = SIGA.s4 - SIGA.sc;
	data[clipSev+1] = SIGA.s5 - SIGA.sd;
	data[clipEig+0] = SIGA.s6 - SIGA.se; 
	data[clipEig+1] = SIGA.s7 - SIGA.sf;
	#endif
	#if 0
	data[clipOne+0] = 0;//
	data[clipOne+1] = 0;
	data[clipTwo+0] = coeffUse;
	data[clipTwo+1] = coeffUse;
	data[clipThr+0] = 2*coeffUse;
	data[clipThr+1] = 2*coeffUse;
	data[clipFou+0] = 3*coeffUse;
	data[clipFou+1] = 3*coeffUse;
	data[clipFiv+0] = 4*coeffUse;//
	data[clipFiv+1] = 4*coeffUse;
	data[clipSix+0] = 5*coeffUse;
	data[clipSix+1] = 5*coeffUse;
	data[clipSev+0] = 111;
	data[clipSev+1] = kIndex;
	data[clipEig+0] = clSet7.x;
	data[clipEig+1] = clSet7.y;
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

