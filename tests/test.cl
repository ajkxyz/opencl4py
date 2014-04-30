__kernel __attribute__((vec_type_hint(float4)))
void test(__global float *a, __global float *b, const float c) {
  size_t i = get_global_id(0);
  a[i] += b[i] * c;
}
