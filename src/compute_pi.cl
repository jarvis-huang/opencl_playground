void kernel compute_pi(global float* global_sum, local float* local_sum,
                       const float step, const int ninput_per_group) {
  int igrp = get_group_id(0);
  int iloc = get_local_id(0);
  int nloc = get_local_size(0);
  int ngrp = get_num_groups(0);

  // SIZE == total problem size
  int start = ninput_per_group * igrp;

  float temp = 0.0;
  float x = 0.0;
  int k = 0;
  for (k = iloc; k < ninput_per_group; k += nloc) {
    x = (start + k + 0.5) * step;
    temp += 4.0 / (1 + x * x);
  }
  local_sum[iloc] = temp;

  // if (igrp == 0 && iloc < 10) {
  //   printf("group[%d][%d]=%.2f, nloc=%d, start=%d, k=%d\n", igrp, iloc, temp,
  //          nloc, start, k);
  // }
  barrier(CLK_LOCAL_MEM_FENCE);

  temp = 0.0;
  for (int j = 0; j < nloc; j++) {
    temp += local_sum[j];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // if (igrp < 10) printf("group[%d]=%.2f\n", igrp, temp * step);
  global_sum[igrp] = temp;
}
