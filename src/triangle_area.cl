void kernel triangle_area(global float* C, const float left, const float up, const float right) {   
  //int i = get_global_id(0);
  //int n = get_global_size(0); // total num of WI
  int i = get_local_id(0);
  int n = get_local_size(0); // total num of WI
  float step = up / n;

  /* Compute left and right end points for this WI */
  float y = i * step;
  float bound_left = (up - y) / up * left;
  float bound_right = (up - y) / up * right;
  C[i] = (bound_left + bound_right) * step;

  barrier(CLK_GLOBAL_MEM_FENCE);

  /* Reduce (into 10 groups) */
  if (i<100) {
    float sum1 = 0;
    for (int j=i; j<n; j+=100) {
      sum1 += C[j];
    }
    C[i] = sum1;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  /* Reduce (into one number) */
  if (i==0) {
    float sum2 = 0;
    for (int j=0; j<100; j++) {
      sum2 += C[j];
    }
    C[i] = sum2;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

}
