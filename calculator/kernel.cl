kernel void vectorAdd(global float *out, global float *a, global float *b)
{
	size_t i = get_global_id(0);
	out[i] = a[i] + b[i];
}

kernel void vectorDot(global float *out, global float *a, global float *b)
{
	size_t gid = get_global_id(0);
	out[gid] = a[gid] * b[gid];
}

kernel void matrixMultiply(constant int *pR, constant int *pM, constant int *pC, global float *out, global float *x, global float *y)
{
	int R = *pR, M = *pM, C = *pC;

	// global id (gid) points (r,c)-th element of out, where gid == r * C + c
	size_t gid = get_global_id(0);
	int r = gid / C, c = gid % C;

	out[gid] = 0;
	for (int i = 0; i < M; ++i) {
		out[gid] += x[r*M+i] * y[i*C+c];
	}
}

kernel void vectorElementWiseGreaterThan(global float *out, global float *x, global float *y)
{
	size_t gid = get_global_id(0);
	out[gid] = x[gid] > y[gid] ? 1.0 : 0;
}

kernel void vectorSigmoid(global float *out, global float *x)
{
	size_t gid = get_global_id(0);
	out[gid] = 1.0 / (1 + exp(-x[gid]));
}

kernel void vectorReLU(global float *out, global float *x)
{
	size_t gid = get_global_id(0);
	out[gid] = max((float)0, x[gid]);
}
