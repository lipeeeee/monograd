__kernel void reduce_{op}_{shape}(
    __global const {type}* in,
    __global {type}* out,
    __local {type}* scratch,
    const int n                    
) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int g_stride = get_global_size(0);
    int l_size = get_local_size(0);

    {type} acc = {identity};

    for (int i = gid; i < n; i += g_stride) {
        acc = {combine}(acc, in[i]);
    }

    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int s = l_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = {combine}(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (lid == 0) {
        int group_id = get_group_id(0);
        out[group_id] = scratch[0];
    }
}
