__kernel void radix_sort(__global int* input, __global int* output, __global int* histogram, const int bit, const int size) {
    int tid = get_global_id(0);
    int bit_value = (input[tid] >> bit) & 1;

    __local int local_histogram[2];
    if (get_local_id(0) < 2) local_histogram[get_local_id(0)] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_inc(&local_histogram[bit_value]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < 2) atomic_add(&histogram[get_local_id(0)], local_histogram[get_local_id(0)]);
    barrier(CLK_GLOBAL_MEM_FENCE);

    __local int prefix_sum[2];
    if (get_local_id(0) == 0) {
        prefix_sum[0] = 0;
        prefix_sum[1] = histogram[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int position;
    if (bit_value == 0) position = prefix_sum[0] + get_local_id(0);
    else position = prefix_sum[1] + get_local_id(0);

    output[position] = input[tid];
}
