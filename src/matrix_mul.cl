void kernel matrix_mul(global const int* A, global const int* B, global int* C, int N) {
    size_t i = get_global_id(0);
    // A * B => C
    // A, B and C are NxN row-major matrices
    // each thread only responsible for one row of output matrix C
    for (size_t j = 0; j<N; j++) { // loop over each location of output matrix C
        int s = 0;
        for (size_t k = 0; k<N; k++) { // inner loop to compute vector dot product
            // A[i, k] * B[k, j]
            s += A[i*N+k] * B[k*N+j];
        }
        C[i*N+j] = s;
    }
}
