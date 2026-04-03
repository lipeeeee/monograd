__kernel void reduce_{op}_{shape}(
    __global const {type}* in,
    __global {type}* out,
    const int n,
    const int reduce_size
) {
    int gid = get_global_id(0);
    if (gid >= n) return;

    {type} acc = {identity};
    for (int k = 0; k < reduce_size; k++) {
        acc = {combine}(acc, in[{input_index}]);
    }
    out[gid] = acc;
}
