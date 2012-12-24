#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define CLPI 3.141592653589793238462643383279 // acos(-1)
#define CLPT 6.283185307179586476925286766559 // acos(-1)*2
#define d707 0.707106781186547524400844362104 // cos(acos(-1)/4)

__kernel void twiddles(__global double2 *twiddle, int size)
{
	#if 1
	int idX = get_global_id(0);
	twiddle[idX] = (double2)(cos(CLPT*idX/size),-sin(CLPT*idX/size));
	#endif
}

__kernel void DIT8C2C(	__global double *data, 
						__global double2 *twiddle,
						const int x, const int y, const int z,
						const int stage, const int dir, const int type ) 
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);
	int idZ = get_global_id(2);

	#if 1
	int powMaxLvl = 4;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int xx;
	for (xx = 0; xx < powLevels; xx++) {
		powX *= pow(8.0f,powMaxLvl);
	}
	powX *= pow(8.0f,powRemain);
	powXm1 = powX/8;
	#endif
	#if 0 // Rounding error appearing only here !!!!  
	int powX = exp2(log2(8.)*stage);
	int powXm1 = powX/8;
	#endif

	int BASE 	= 0;
	int STRIDE 	= 1;

	int yIndex, kIndex, red, coeffUse;
	
	switch(type)
	{
		case 1: BASE 		= idZ*x*y + idY*x; 
				yIndex 		= idX / powXm1;
				kIndex 		= idX % powXm1;
				red 		= x / 4;	
				coeffUse 	= kIndex * (x / powX);
				break;
		case 2: BASE 		= idZ*x*y + idX; 
				STRIDE 		= x; 
				yIndex 		= idY / powXm1;
				kIndex 		= idY % powXm1;
				red 		= y / 4;
				coeffUse 	= kIndex * (y / powX);
				break;
		case 3: BASE 		= idY*x + idX; 
				STRIDE 		= x * y; 
				yIndex 		= idZ / powXm1;
				kIndex 		= idZ % powXm1;
				red 		= z / 4;
				coeffUse 	= kIndex * (z / powX);
				break; 
	}

	int clipOne 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 0 * powXm1));
	int clipTwo 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 1 * powXm1));
	int clipThr		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 2 * powXm1));
	int clipFou		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 3 * powXm1));
	int clipFiv		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 4 * powXm1));
	int clipSix		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 5 * powXm1));
	int clipSev		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 6 * powXm1));
	int clipEig		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 7 * powXm1));

	double16 SIGA = (double16)(	data[clipOne+0],data[clipOne+1],	// s0, s1
								data[clipTwo+0],data[clipTwo+1],	// s2, s3
								data[clipThr+0],data[clipThr+1],	// s4, s5
								data[clipFou+0],data[clipFou+1],	// s6, s7
								data[clipFiv+0],data[clipFiv+1],	// s8, s9
								data[clipSix+0],data[clipSix+1],	// sa, sb
								data[clipSev+0],data[clipSev+1],	// sc, sd
								data[clipEig+0],data[clipEig+1]);	// se, sf

	#if 1
	double2 clSet1,TMP;
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

	#endif
}

__kernel void DIT4C2C(	__global double *data, 
						__global double2 *twiddle,
						const int x, const int y, const int z,
						const int stage, const int dir, const int type ) 
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);
	int idZ = get_global_id(2);

	#if 1
	int powMaxLvl = 7;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int xx;
	for (xx = 0; xx < powLevels; xx++) {
		powX *= pow(4.0f,powMaxLvl);
	}
	powX *= pow(4.0f,powRemain);
	powXm1 = powX/4;
	#endif
	#if 0
	int powX = exp2(log2(4.)*stage);
	int powXm1 = powX/4;
	#endif

	int clipOne, clipTwo, clipThr, clipFou, yIndex, kIndex, red, coeffUse;
	
	int BASE 	= 0;
	int STRIDE 	= 1;

	switch(type)
	{
		case 1: BASE = idZ*x*y + idY*x; 
				yIndex 		= idX / powXm1;
				kIndex 		= idX % powXm1;
				red 		= x / 4;	
				coeffUse 	= kIndex * (x / powX);
				break;
		case 2: BASE 		= idZ*x*y + idX; 
				STRIDE 		= x; 
				yIndex 		= idY / powXm1;
				kIndex 		= idY % powXm1;
				red 		= y / 4;
				coeffUse 	= kIndex * (y / powX);
				break;
		case 3: BASE 		= idY*x + idX; 
				STRIDE 		= x * y; 
				yIndex 		= idZ / powXm1;
				kIndex 		= idZ % powXm1;
				red 		= z / 4;
				coeffUse 	= kIndex * (z / powX);
				break; 
	}

	clipOne 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 0 * powXm1));
	clipTwo 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 1 * powXm1));
	clipThr		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 2 * powXm1));
	clipFou		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 3 * powXm1));

	double8 SIGA = (double8)(	data[clipOne+0],	data[clipOne+1],
								data[clipTwo+0],	data[clipTwo+1],
								data[clipThr+0],	data[clipThr+1],
								data[clipFou+0],	data[clipFou+1]	);


	int quad, buad;
	double2 TEMPC, clSet1;

	#if 1
	quad = coeffUse / red;
	buad = coeffUse % red;
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
	if (kIndex != 0) {
		//clSet2.x = cos(2.*2.*CLPI*kIndex/powX);
		//clSet2.y = -sin(2.*2.*CLPI*kIndex/powX);
		TEMPC.x = SIGA.s2 * clSet2.x - SIGA.s3 * clSet2.y;
		TEMPC.y = SIGA.s3 * clSet2.x + SIGA.s2 * clSet2.y;
		SIGA.s2 = TEMPC.x;
		SIGA.s3 = TEMPC.y;
		//clSet1.x = cos(2.*CLPI*kIndex/powX);
		//clSet1.y = -sin(2.*CLPI*kIndex/powX);
		TEMPC.x = SIGA.s4 * clSet1.x - SIGA.s5 * clSet1.y;
		TEMPC.y = SIGA.s5 * clSet1.x + SIGA.s4 * clSet1.y;
		SIGA.s4 = TEMPC.x;
		SIGA.s5 = TEMPC.y;
		//clSet3.x = cos(3.*2.*CLPI*kIndex/powX);
		//clSet3.y = -sin(3.*2.*CLPI*kIndex/powX);
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


__kernel void DIT2C2C(	__global double *data, 
						__global double2 *twiddle,
						const int x, const int y, const int z,
						const int stage, const int dir, const int type ) 
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);
	int idZ = get_global_id(2);

	int BASE 	= 0;
	int STRIDE 	= 1;

	#if 1
	int powMaxLvl = 11;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int xx;
	for (xx = 0; xx < powLevels; xx++) {
		powX *= pow(2.0f,powMaxLvl);
	}
	powX *= pow(2.0f,powRemain);
	powXm1 = powX/2;
	#endif
	#if 0
	int powX = exp2(log2(2.)*stage);
	int powXm1 = powX/2;
	#endif

	int yIndex, kIndex, clipOne, clipTwo, coeffUse, red, quad, buad;
	double2 clSet1;
	
	switch(type)
	{
		case 1: BASE 		= idZ*x*y + idY*x; 
				yIndex 		= idX / powXm1;
				kIndex 		= idX % powXm1;	
				coeffUse 	= kIndex * (x / powX);
				red 		= x / 4;
				break;
		case 2: BASE 		= idZ*x*y + idX; 
				STRIDE 		= x; 
				yIndex 		= idY / powXm1;
				kIndex 		= idY % powXm1;
				coeffUse 	= kIndex * (y / powX);
				red 		= y / 4;
				break;
		case 3: BASE 		= idY*x + idX; 
				STRIDE 		= x * y; 
				yIndex 		= idZ / powXm1;
				kIndex 		= idZ % powXm1;
				coeffUse 	= kIndex * (z / powX);
				red  		= z / 4;
				break; 
	}

	clipOne 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX));
	clipTwo 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + powXm1));
	
	quad = coeffUse/red;
	buad = coeffUse%red;

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

	double4 LOC = (double4)(	data[clipOne + 0],	data[clipOne + 1],
								data[clipTwo + 0],		data[clipTwo + 1]);
	double4 FIN = (double4)(	LOC.x + LOC.z * clSet1.x - LOC.w * clSet1.y,
								LOC.y + LOC.w * clSet1.x + LOC.z * clSet1.y,
								LOC.x - LOC.z * clSet1.x + LOC.w * clSet1.y,
								LOC.y - LOC.w * clSet1.x - LOC.z * clSet1.y);

	data[clipOne + 0] = FIN.x;
	data[clipOne + 1] = FIN.y;
	data[clipTwo + 0] 	= FIN.z;
	data[clipTwo + 1] 	= FIN.w;

}

__kernel void divide(	__global double2 *data, int x, int y, int z)
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);
	int idZ = get_global_id(2);
	data[idZ*x*y+idY*x+idX] /= (x*y*z);
}

__kernel void swapkernel(	__global double *data,	// initial data
						const int x,	// dims
						const int y,
						const int z,
						__global int *bitX,	// bitrev data
						__global int *bitY,
						__global int *bitZ,
						const unsigned int type) // x or y or z
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);
	int idZ = get_global_id(2);
	__private int BASE = 0;
	__private int STRIDE = 1;
	__private double holder;
	__private int runner = 0;
	__private int OLD = 0, NEW = 0;

	switch(type)
	{
		case 1: BASE = idZ*x*y + idY*x;
				if (idX < bitX[idX]) {
					OLD = 2*(BASE+STRIDE*idX);
					NEW = 2*(BASE+STRIDE*bitX[idX]);

					holder = data[NEW];
					data[NEW] = data[OLD];
					data[OLD] = holder;

					holder = data[NEW+1];
					data[NEW+1] = data[OLD+1];
					data[OLD+1] = holder;
				}
				break;
		case 2: BASE = idZ*x*y + idX; STRIDE = x; 
				if (idY < bitY[idY]) {
					OLD = 2*(BASE+STRIDE*idY);
					NEW = 2*(BASE+STRIDE*bitY[idY]);

					holder = data[NEW];
					data[NEW] = data[OLD];
					data[OLD] = holder;

					holder = data[NEW+1];
					data[NEW+1] = data[OLD+1];
					data[OLD+1] = holder;
				}
				break;
		case 3: BASE = idY*x + idX; STRIDE = x * y;
				if (idZ < bitZ[idZ]) {
					OLD = 2*(BASE+STRIDE*idZ);
					NEW = 2*(BASE+STRIDE*bitZ[idZ]);

					holder = data[NEW];
					data[NEW] = data[OLD];
					data[OLD] = holder;

					holder = data[NEW+1];
					data[NEW+1] = data[OLD+1];
					data[OLD+1] = holder;
				}
				break;
	}
}

__kernel void reverse(__global int *bitRev, int logSize)
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

