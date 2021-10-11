void kernel matrix_mul(global const int* A, global const int* B, global int* C, const int N) {
    size_t i = get_global_id(0);

    // Save a row of A into private memory
    int temp;
    int Awrk[1024]; // make sure N<=1024 or else you will get SEG FAULT!
    for (size_t k=0; k<N; k++)
        Awrk[k] = A[i*N+k];

    // A * B => C
    // A, B and C are NxN row-major matrices
    // Each thread responsible for computing one row of output matrix C
    for (size_t j = 0; j<N; j++) {
        temp = 0;
        for (size_t k = 0; k<N; k++) { // inner loop to compute vector dot product
            // A[i, k] * B[k, j]
            temp += Awrk[k] * B[k*N+j];
        }
        C[i*N+j] = temp;
    }
}
