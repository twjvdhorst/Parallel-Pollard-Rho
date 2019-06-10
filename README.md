# Parallel Pollard Rho

This project implements Pollard's Rho method for solving discrete logarithm problems on the GPU.

The program can handle groups Z/pZ, with p an odd number. The chosen element g must be of prime order q.
The program finds an integer 0 <= x <= q - 1, for which g^x = y, with y in <g>.

The application is written in C#, using OpenCL and Cloo for using the GPU.
