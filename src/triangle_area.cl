void kernel triangle_area(global float* C, local float* local_mem, const float left, const float up, const float right) {   
  int gid = get_global_id(0);
  int g_size = get_global_size(0); // total num of WI
  int lid = get_local_id(0);
  int wg_size = get_local_size(0); // num of WI in a WG
  int gpid = get_group_id(0);
  
  /* Compute left and right end points for this WI */
  float step = up / g_size; // each thread responsible for a strip of this height
  float y = gid * step;
  float bound_left = (up - y) / up * left;
  float bound_right = (up - y) / up * right;
  local_mem[lid] = (bound_left + bound_right) * step; // save strip area into local mem
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Reduce within each WG -> global mem */
  if (lid==0) {
    float sum1 = 0;
    for (int j=0; j<wg_size; j++) {
      sum1 += local_mem[j];
    }
    //local_mem[0] = sum1;
    C[gpid] = sum1;
  }
}
