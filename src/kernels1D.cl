#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define CLPI 3.141592653589793238462643383279 // acos(-1)
#define CLPT 6.283185307179586476925286766559 // acos(-1)*2
#define d707 0.707106781186547524400844362104 // cos(acos(-1)/4)

__kernel void twid1D(__global double2 *twiddle, int size)
{
	#if 1
	int idX = get_global_id(0);
	twiddle[idX] =  (double2)(cos(CLPT*idX/size),-sin(CLPT*idX/size));
	#endif
}

__kernel void DIT4C2C(	__global double *data, 
						__global double2 *twiddle,
						const int size, unsigned int stage,
						unsigned int dir ) 
{

	int idX = get_global_id(0);
	
	#if 1
	int powMaxLvl = 7;
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
	#endif
	#if 0
	int powX = exp2(log2(4.)*stage);
	int powXm1 = powX/4;
	#endif

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

	int coeffUse = kIndex * (size / powX);
	int red = size/4;
	double2 clSet1;

	#if 1
	int quad = coeffUse / red;
	int buad = coeffUse % red;
	switch(quad) {
		case 0:	clSet1 = (double2)( twiddle[buad].x,  twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y, -twiddle[buad].x); break;
		case 2:	clSet1 = (double2)(-twiddle[buad].x, -twiddle[buad].y); break;
		case 3:	clSet1 = (double2)(-twiddle[buad].y,  twiddle[buad].x); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TEMPC.x = SIGA.s4 * clSet1.x - SIGA.s5 * clSet1.y;
		TEMPC.y = SIGA.s5 * clSet1.x + SIGA.s4 * clSet1.y;
		SIGA.s4 = TEMPC.x;
		SIGA.s5 = TEMPC.y;
	}

	quad = (2*coeffUse) / red;
	buad = (2*coeffUse) % red;
	switch(quad) {
		case 0:	clSet1 = (double2)( twiddle[buad].x,  twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y, -twiddle[buad].x); break;
		case 2:	clSet1 = (double2)(-twiddle[buad].x, -twiddle[buad].y); break;
		case 3:	clSet1 = (double2)(-twiddle[buad].y,  twiddle[buad].x); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TEMPC.x = SIGA.s2 * clSet1.x - SIGA.s3 * clSet1.y;
		TEMPC.y = SIGA.s3 * clSet1.x + SIGA.s2 * clSet1.y;
		SIGA.s2 = TEMPC.x;
		SIGA.s3 = TEMPC.y;
	}

	quad = (3*coeffUse) / red;
	buad = (3*coeffUse) % red;
	switch(quad) {
		case 0:	clSet1 = (double2)( twiddle[buad].x,  twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y, -twiddle[buad].x); break;
		case 2:	clSet1 = (double2)(-twiddle[buad].x, -twiddle[buad].y); break;
		case 3:	clSet1 = (double2)(-twiddle[buad].y,  twiddle[buad].x); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {	
		TEMPC.x = SIGA.s6 * clSet1.x - SIGA.s7 * clSet1.y;
		TEMPC.y = SIGA.s7 * clSet1.x + SIGA.s6 * clSet1.y;
		SIGA.s6 = TEMPC.x;
		SIGA.s7 = TEMPC.y;
	}
	#endif
	#if 0
	double2 clSet2, clSet3;
	if (kIndex != 0) {
		clSet2.x =  cos(2.*CLPT*kIndex/powX);
		clSet2.y = -sin(2.*CLPT*kIndex/powX);
		TEMPC.x = SIGA.s2 * clSet2.x - SIGA.s3 * clSet2.y;
		TEMPC.y = SIGA.s3 * clSet2.x + SIGA.s2 * clSet2.y;
		SIGA.s2 = TEMPC.x;
		SIGA.s3 = TEMPC.y;
		clSet1.x = cos(CLPT*kIndex/powX);
		clSet1.y = -sin(CLPT*kIndex/powX);
		TEMPC.x = SIGA.s4 * clSet1.x - SIGA.s5 * clSet1.y;
		TEMPC.y = SIGA.s5 * clSet1.x + SIGA.s4 * clSet1.y;
		SIGA.s4 = TEMPC.x;
		SIGA.s5 = TEMPC.y;
		clSet3.x = cos(3.*CLPT*kIndex/powX);
		clSet3.y = -sin(3.*CLPT*kIndex/powX);
		TEMPC.x = SIGA.s6 * clSet3.x - SIGA.s7 * clSet3.y;
		TEMPC.y = SIGA.s7 * clSet3.x + SIGA.s6 * clSet3.y;
		SIGA.s6 = TEMPC.x;
		SIGA.s7 = TEMPC.y;
	}	
	#endif
	
	if (dir == 1) {
		data[clipOne+0] = SIGA.s0 + SIGA.s2 + SIGA.s4 + SIGA.s6;
		data[clipOne+1] = SIGA.s1 + SIGA.s3 + SIGA.s5 + SIGA.s7;
		data[clipTwo+0] = SIGA.s0 - SIGA.s2 + SIGA.s5 - SIGA.s7;
		data[clipTwo+1] = SIGA.s1 - SIGA.s3 - SIGA.s4 + SIGA.s6;
		data[clipThr+0] = SIGA.s0 + SIGA.s2 - SIGA.s4 - SIGA.s6;
		data[clipThr+1] = SIGA.s1 + SIGA.s3 - SIGA.s5 - SIGA.s7;
		data[clipFou+0] = SIGA.s0 - SIGA.s2 - SIGA.s5 + SIGA.s7;
		data[clipFou+1] = SIGA.s1 - SIGA.s3 + SIGA.s4 - SIGA.s6;
	}
	else if (dir == 0) {
		data[clipOne+0] = SIGA.s0 + SIGA.s2 + SIGA.s4 + SIGA.s6;
		data[clipOne+1] = SIGA.s1 + SIGA.s3 + SIGA.s5 + SIGA.s7;
		data[clipTwo+0] = SIGA.s0 - SIGA.s2 - SIGA.s5 + SIGA.s7;
		data[clipTwo+1] = SIGA.s1 - SIGA.s3 + SIGA.s4 - SIGA.s6;
		data[clipThr+0] = SIGA.s0 + SIGA.s2 - SIGA.s4 - SIGA.s6;
		data[clipThr+1] = SIGA.s1 + SIGA.s3 - SIGA.s5 - SIGA.s7;
		data[clipFou+0] = SIGA.s0 - SIGA.s2 + SIGA.s5 - SIGA.s7;
		data[clipFou+1] = SIGA.s1 - SIGA.s3 - SIGA.s4 + SIGA.s6;
	}
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
						__global double2 *twiddle,
						const int size, unsigned int stage,
						unsigned int dir ) 
{
	int idX = get_global_id(0);

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

	
	int coeffUse = kIndex * (size / powX);	
	int red = size/4;
	double2 clSet1;

	int quad = coeffUse/red;
	int buad = coeffUse%red;
	switch(quad) {
		case 0: clSet1 = (double2)(	twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)(	twiddle[buad].y,	-twiddle[buad].x); break;
		case 2:	clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.s8 * clSet1.x - SIGA.s9 * clSet1.y;
		TMP.y = SIGA.s8 * clSet1.y + SIGA.s9 * clSet1.x;
		SIGA.s8 = TMP.x;
		SIGA.s9 = TMP.y;
	}	

	quad = (2*coeffUse)/red;
	buad = (2*coeffUse)%red;
	switch(quad) {
		case 0: clSet1 = (double2)( twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y,	-twiddle[buad].x); break;
		case 2: clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.s4 * clSet1.x - SIGA.s5 * clSet1.y;
		TMP.y = SIGA.s4 * clSet1.y + SIGA.s5 * clSet1.x;
		SIGA.s4 = TMP.x;
		SIGA.s5 = TMP.y;
	}	

	quad = (3*coeffUse)/red;
	buad = (3*coeffUse)%red;
	switch(quad) {
		case 0: clSet1 = (double2)( twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y,	-twiddle[buad].x); break;
		case 2: clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.sc * clSet1.x - SIGA.sd * clSet1.y;
		TMP.y = SIGA.sc * clSet1.y + SIGA.sd * clSet1.x;
		SIGA.sc = TMP.x;
		SIGA.sd = TMP.y;
	}	

	quad = (4*coeffUse)/red;
	buad = (4*coeffUse)%red;
	switch(quad) {
		case 0: clSet1 = (double2)( twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y,	-twiddle[buad].x); break;
		case 2: clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.s2 * clSet1.x - SIGA.s3 * clSet1.y;
		TMP.y = SIGA.s2 * clSet1.y + SIGA.s3 * clSet1.x;
		SIGA.s2 = TMP.x;
		SIGA.s3 = TMP.y;
	}

	quad = (5*coeffUse)/red;
	buad = (5*coeffUse)%red;
	switch(quad) {
		case 0: clSet1 = (double2)( twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y,	-twiddle[buad].x); break;
		case 2: clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.sa * clSet1.x - SIGA.sb * clSet1.y;
		TMP.y = SIGA.sa * clSet1.y + SIGA.sb * clSet1.x;
		SIGA.sa = TMP.x;
		SIGA.sb = TMP.y;
	}

	quad = (6*coeffUse)/red;
	buad = (6*coeffUse)%red;
	switch(quad) {
		case 0: clSet1 = (double2)( twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y,	-twiddle[buad].x); break;
		case 2: clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.s6 * clSet1.x - SIGA.s7 * clSet1.y;
		TMP.y = SIGA.s6 * clSet1.y + SIGA.s7 * clSet1.x;
		SIGA.s6 = TMP.x;
		SIGA.s7 = TMP.y;
	}	

	quad = (7*coeffUse)/red;
	buad = (7*coeffUse)%red;
	switch(quad) {
		case 0: clSet1 = (double2)( twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y,	-twiddle[buad].x); break;
		case 2: clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
		case 3: clSet1 = (double2)(-twiddle[buad].y,	 twiddle[buad].x); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.se * clSet1.x - SIGA.sf * clSet1.y;
		TMP.y = SIGA.se * clSet1.y + SIGA.sf * clSet1.x;
		SIGA.se = TMP.x;
		SIGA.sf = TMP.y;
	}	

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

	if (dir == 1) {
		TMP = (double2)((SIGB.sa + SIGB.sb)*d707, (SIGB.sf - SIGB.se)*d707);
		SIGB.sb = (SIGB.sb - SIGB.sa) * d707;
		SIGB.sf = (SIGB.sf + SIGB.se) * -d707;
		SIGB.sa = TMP.x;	
		SIGB.se = TMP.y;
		TMP.x	= SIGB.s7; SIGB.s7 = -SIGB.s6; SIGB.s6 = TMP.x; 
	}
	else if (dir == 0) {
		TMP = (double2)((SIGB.sa - SIGB.sb)*d707, (SIGB.sf + SIGB.se)*-d707);
		SIGB.sb = (SIGB.sb + SIGB.sa) * d707;
		SIGB.sf = (SIGB.se - SIGB.sf) * d707;
		SIGB.sa = TMP.x;
		SIGB.se = TMP.y;
		TMP.x 	= -SIGB.s7; SIGB.s7 = SIGB.s6; SIGB.s6 = TMP.x;
	}

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
	if (dir == 1) {
		SIGA.sc = SIGB.s9 - SIGB.sd;
		SIGA.sd = SIGB.sc - SIGB.s8;
		SIGA.se = SIGB.sb - SIGB.sf;
		SIGA.sf = SIGB.se - SIGB.sa;
	}
	else if (dir == 0) {
		SIGA.sc = SIGB.sd - SIGB.s9;
		SIGA.sd = SIGB.s8 - SIGB.sc;
		SIGA.se = SIGB.sf - SIGB.sb;
		SIGA.sf = SIGB.sa - SIGB.se;
	}

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
	data[clipTwo+0] = 0;
	data[clipTwo+1] = 0;
	data[clipThr+0] = 0;
	data[clipThr+1] = 0;
	data[clipFou+0] = 0;
	data[clipFou+1] = 0;
	data[clipFiv+0] = 0;//
	data[clipFiv+1] = 0;
	data[clipSix+0] = 0;
	data[clipSix+1] = 0;
	data[clipSev+0] = 0;
	data[clipSev+1] = 0;
	data[clipEig+0] = idZ;
	data[clipEig+1] = 0;
	#endif
	#if 0
	data[2*(idZ*x*y+idY*x+idX)] = powX;
	data[2*(idZ*x*y+idY*x+idX)+1] = 0;
	#endif
}

__kernel void DIT2C2C(	__global double *data, 
						__global double2 *twiddle,
						const int size, unsigned int stage,
						unsigned int dir ) 
{
	#if 1
	int idX = get_global_id(0);

	#if 1
	int powMaxLvl = 11;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int x;
	for (x = 0; x < powLevels; x++)	powX *= pow(2.0f,powMaxLvl);
	powX *= pow(2.0f,powRemain);
	powXm1 = powX/2;
	#endif
	#if 0
	int powX = exp2(log2(2.)*stage);
	int powXm1 = powX/2;
	#endif

	int yIndex = idX / powXm1;
	int kIndex = idX % powXm1;

	int clipStart 	= 2*(kIndex + yIndex * powX);
	int clipEnd 	= 2*(kIndex + yIndex * powX + powXm1);
	int coeffUse 	= kIndex * (size/powX);

	int red = size/4;
	double2 clSet1;

	int quad = coeffUse/red;
	int buad = coeffUse%red;

	switch(quad) {
		case 0:	clSet1 = (double2)( twiddle[buad].x,  twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y, -twiddle[buad].x); break;
		case 2:	clSet1 = (double2)(-twiddle[buad].x, -twiddle[buad].y); break;
		case 3:	clSet1 = (double2)(-twiddle[buad].y,  twiddle[buad].x); break;
	}
	if (dir == 0) clSet1.y *= -1;
	#if 0
	clSet1.x 	= cos(two*CLPI*(coeffUse/2)/xR);
	clSet1.y 	= sin(two*CLPI*(coeffUse/2)/xR);
	#endif

	double4 LOC = (double4)(	data[clipStart + 0],	data[clipStart + 1],
								data[clipEnd + 0],		data[clipEnd + 1]);
	double4 FIN = (double4)(	LOC.x + LOC.z * clSet1.x - LOC.w * clSet1.y,
								LOC.y + LOC.w * clSet1.x + LOC.z * clSet1.y,
								LOC.x - LOC.z * clSet1.x + LOC.w * clSet1.y,
								LOC.y - LOC.w * clSet1.x - LOC.z * clSet1.y);

	data[clipStart + 0] = FIN.x;
	data[clipStart + 1] = FIN.y;
	data[clipEnd + 0] 	= FIN.z;
	data[clipEnd + 1] 	= FIN.w;
	#if 0	// Debug
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

__kernel void divide1D(	__global double2 *data, const int size)
{
	int idX = get_global_id(0);
	data[idX] /= size;
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
	bitRev[global_id+get_global_size(0)] = sum+1;
}

