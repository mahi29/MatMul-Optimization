README
*************
Given a naive implementation of matrix multiplication, optimized and increased performance of 50 fold (1 GFlop/s to 50 GFlop/s). This increase was gained by using SSE Instructions along with register and cache blocking. Also used OpenMP to implement parallelization, along with loop reordering, to further increase speed. 