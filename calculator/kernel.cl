kernel void vec_add(global float *out, global float *a, global float *b)
{
	size_t i = get_global_id(0);
	out[i] = a[i] + b[i];
}

kernel void vec_dot(global float *out, global float *a, global float *b)
{
	size_t gid = get_global_id(0);
	out[gid] = a[gid] * b[gid];
}
