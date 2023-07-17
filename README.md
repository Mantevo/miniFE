# miniFE Finite Element Mini-Application

MiniFE is an proxy application for unstructured implicit finite element codes.
It is similar to HPCCG and pHPCCG but provides a much more complete vertical 
covering of the steps in this class of applications. MiniFE also provides 
support for computation on multicore nodes, including pthreads and Intel 
Threading Building Blocks (TBB) for homogeneous multicore and CUDA for GPUs. 
Like HPCCG and pHPCCG, MiniFE is intended to be the "best approximation to an
unstructured implicit finite element or finite volume application, but in 8000 lines or fewer."

## License Information

MiniFE is licensed under the LGPL-3. See LICENSE for more information.

## Additional Information

Please see the README.FIRST which accompanies the MiniFE source code.

For more details about miniFE and a comparison with an application code, please see the following reference:

P.T. Lin, M.A. Heroux, R.F. Barrett, and A.B. Williams, "Assessing a mini-application as a performance proxy for a finite element method engineering application," Concurrency and Computation: Practice and Experience, 27(17):5374–5389, 2015. DOI: 10.1002/cpe.3587
