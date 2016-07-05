#pragma once
#include <CudaUtils.h>

namespace miniFE {

  namespace Hex8 {

#ifdef __CUDACC__
    __constant__ MINIFE_SCALAR gauss_pts_c[numGaussPointsPerDim];
#endif

    template<typename Scalar>
      __device__ void compute_detJ_invJ_grad_vals(int elethidx, const Scalar elemNodeCoords[Hex8::spatialDim],
          Scalar invJ_grad_vals[numNodesPerElem*spatialDim],
          Scalar& detJ, const Scalar *gradients)
      {
        Scalar J0=0,J1=0,J2=0,J3=0,J4=0,J5=0,J6=0,J7=0,J8=0;
        Scalar grad_vals0, grad_vals1, grad_vals2;
#pragma unroll
        for(size_t i=0; i<numNodesPerElem; ++i) {

          int texidx = elethidx*spatialDim*numNodesPerElem+i*spatialDim;
          grad_vals0=__ldg(gradients + texidx+0);
          grad_vals1=__ldg(gradients + texidx+1);
          grad_vals2=__ldg(gradients + texidx+2);

          Scalar elemNodeCoords0=elemNodeCoords[i*Hex8::spatialDim];
          Scalar elemNodeCoords1=elemNodeCoords[i*Hex8::spatialDim+1];
          Scalar elemNodeCoords2=elemNodeCoords[i*Hex8::spatialDim+2];

          J0 += grad_vals0*elemNodeCoords0;
          J1 += grad_vals0*elemNodeCoords1;
          J2 += grad_vals0*elemNodeCoords2;

          J3 += grad_vals1*elemNodeCoords0;
          J4 += grad_vals1*elemNodeCoords1;
          J5 += grad_vals1*elemNodeCoords2;

          J6 += grad_vals2*elemNodeCoords0;
          J7 += grad_vals2*elemNodeCoords1;
          J8 += grad_vals2*elemNodeCoords2;
        }

        Scalar term0 = J8*J4 - J7*J5;
        Scalar term1 = J8*J1 - J7*J2;
        Scalar term2 = J5*J1 - J4*J2;

        Scalar term3 = J8*J3 - J6*J5;
        Scalar term4 = J8*J0 - J6*J2;
        Scalar term5 = J5*J0 - J3*J2;

        Scalar term6 = J7*J3 - J6*J4;
        Scalar term7 = J7*J0 - J6*J1;
        Scalar term8 = J4*J0 - J3*J1;

        detJ = J0*term0 - J3*term1 + J6*term2;
        Scalar inv_detJ = 1.0/detJ;

        J0 =  term0*inv_detJ;
        J1 = -term1*inv_detJ;
        J2 =  term2*inv_detJ;

        J3 = -term3*inv_detJ;
        J4 =  term4*inv_detJ;
        J5 = -term5*inv_detJ;

        J6 =  term6*inv_detJ;
        J7 = -term7*inv_detJ;
        J8 =  term8*inv_detJ;

#pragma unroll
        for(int j=0; j<numNodesPerElem; ++j) {
          int texidx = elethidx*spatialDim*numNodesPerElem+j*spatialDim;
          Scalar gv0=__ldg(gradients+texidx+0);
          Scalar gv1=__ldg(gradients+texidx+1);
          Scalar gv2=__ldg(gradients+texidx+2);

          invJ_grad_vals[j*spatialDim+0] = J0 * gv0 + J1 * gv1 + J2 * gv2;
          invJ_grad_vals[j*spatialDim+1] = J3 * gv0 + J4 * gv1 + J5 * gv2;
          invJ_grad_vals[j*spatialDim+2] = J6 * gv0 + J7 * gv1 + J8 * gv2;
        }
      }

    template<typename Scalar>
      __device__ __inline__ void compute_detJ(int elethidx, const Scalar elemNodeCoords[Hex8::spatialDim],
          Scalar& detJ, const Scalar *gradients)
      {
        Scalar J0=0,J1=0,J2=0,J3=0,J4=0,J5=0,J6=0,J7=0,J8=0;
        Scalar grad_vals0, grad_vals1, grad_vals2;

#pragma unroll
        for(size_t i=0; i<Hex8::numNodesPerElem; ++i) {
          int texidx = elethidx*spatialDim*numNodesPerElem+i*spatialDim;

          grad_vals0=__ldg(gradients+texidx+0);
          grad_vals1=__ldg(gradients+texidx+1);
          grad_vals2=__ldg(gradients+texidx+2);

          Scalar elemNodeCoords0=elemNodeCoords[Hex8::spatialDim*i];
          Scalar elemNodeCoords1=elemNodeCoords[Hex8::spatialDim*i+1];
          Scalar elemNodeCoords2=elemNodeCoords[Hex8::spatialDim*i+2];

          J0 += grad_vals0*elemNodeCoords0;
          J1 += grad_vals0*elemNodeCoords1;
          J2 += grad_vals0*elemNodeCoords2;

          J3 += grad_vals1*elemNodeCoords0;
          J4 += grad_vals1*elemNodeCoords1;
          J5 += grad_vals1*elemNodeCoords2;

          J6 += grad_vals2*elemNodeCoords0;
          J7 += grad_vals2*elemNodeCoords1;
          J8 += grad_vals2*elemNodeCoords2;
        }

        Scalar term0 = J8*J4 - J7*J5;
        Scalar term1 = J8*J1 - J7*J2;
        Scalar term2 = J5*J1 - J4*J2;


        detJ = J0*term0 - J3*term1 + J6*term2;

      }

    template<typename Scalar>
      __device__ __inline__ void diffusion_matrix_device(Scalar elem_mat[Hex8::numNodesPerElem*(Hex8::numNodesPerElem+1)/2], Scalar invJ_grad_vals[numNodesPerElem*spatialDim], Scalar k_detJ_wi_wj_wk)
      {
        int offset = 0;
#pragma unroll
        for(int m=0; m<numNodesPerElem; ++m) {

          Scalar dpsidx_m=invJ_grad_vals[m*spatialDim+0];
          Scalar dpsidy_m=invJ_grad_vals[m*spatialDim+1];
          Scalar dpsidz_m=invJ_grad_vals[m*spatialDim+2];

          elem_mat[offset++] += k_detJ_wi_wj_wk *
            ((dpsidx_m*dpsidx_m) +
             (dpsidy_m*dpsidy_m) +
             (dpsidz_m*dpsidz_m));

#pragma unroll
          for(int n=0; n<numNodesPerElem; ++n) {
            if(n>=m+1) {
              elem_mat[offset++] += k_detJ_wi_wj_wk *
                ((dpsidx_m * invJ_grad_vals[n*spatialDim+0]) +
                 (dpsidy_m * invJ_grad_vals[n*spatialDim+1]) +
                 (dpsidz_m * invJ_grad_vals[n*spatialDim+2]));
            }
          }
        }
      }



    template<typename Scalar>
      __device__  __inline__ void diffusionMatrix_symm(const Scalar elemNodeCoords[Hex8::spatialDim],
          Scalar elem_mat[Hex8::numNodesPerElem*(Hex8::numNodesPerElem+1)/2],
          const Scalar *gradients)
      {
        //Commented-out debugging sanity check: verify that the incoming
        //element-node coordinates really represent the expected element with the
        //expected volume? (Useful for verifying a unit-test with a hard-coded
        //element. Commented out in general.)
        const Scalar zero = 0;
        miniFE::fill(elem_mat, elem_mat+numNodesPerElem*(numNodesPerElem+1)/2, zero);

        //The following nested loop implements equations 3.4.5 and 3.4.7 on page 88
        //of Reddy & Gartling, "The Finite Element Method in Heat Transfer and Fluid
        //Dynamics", 2nd edition,
        //to compute the element diffusion matrix for the steady conduction equation.

        for(size_t ig=0; ig<numGaussPointsPerDim; ++ig) {
          Scalar wi = gauss_pts_c[ig];

          for(size_t jg=0; jg<numGaussPointsPerDim; ++jg) {
            Scalar wi_wj = wi*gauss_pts_c[jg];

            for(size_t kg=0; kg<numGaussPointsPerDim; ++kg) {

              Scalar detJ = 0.0;

              Scalar invJ_grad_vals[numNodesPerElem*spatialDim];

              int elethidx = ig*4+jg*2+kg;

              compute_detJ_invJ_grad_vals(elethidx, elemNodeCoords,
                  invJ_grad_vals,
                  detJ,gradients);

              Scalar wi_wj_wk = wi_wj*gauss_pts_c[kg];
              const Scalar k = 1.0;
              Scalar k_detJ_wi_wj_wk = k*detJ*wi_wj_wk;


              diffusion_matrix_device(elem_mat, invJ_grad_vals, k_detJ_wi_wj_wk);

            }//for kg
          }//for jg
        }//for ig

      }

    template<typename Scalar>
      __device__  __inline__ void sourceVector(const Scalar elemNodeCoords[Hex8::spatialDim],
          Scalar* __restrict__ elem_vec, const Scalar *gradients, const Scalar *psi)
      {
        const Scalar zero = 0;
        miniFE::fill(elem_vec, elem_vec+numNodesPerElem, zero);

        for(size_t ig=0; ig<numGaussPointsPerDim; ++ig) {
          Scalar wi = gauss_pts_c[ig];

          for(size_t jg=0; jg<numGaussPointsPerDim; ++jg) {
            Scalar wj = gauss_pts_c[jg];
            for(size_t kg=0; kg<numGaussPointsPerDim; ++kg) {
              Scalar wk = gauss_pts_c[kg];

              Scalar detJ = 0;
              int elethidx = ig*4+jg*2+kg;
              compute_detJ(elethidx, elemNodeCoords,detJ, gradients);

              Scalar Q = 1.0;
              Scalar term = Q*detJ*wi*wj*wk;


              Scalar psii;
#pragma unroll
              for(int i=0; i<numNodesPerElem; ++i) {
                psii = __ldg(psi+elethidx*numNodesPerElem+i);
                elem_vec[i] += psii*term;
              }

            }
          }
        }
      }

  }
}
