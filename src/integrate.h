#ifndef INTEGRATE_H
#define INTEGRATE_H

#include "orbitals.h"

/** @name Combinatorics Helpers */
///@{
/** @brief Factorial helper. */
double factorial(int n);
/** @brief Double-factorial helper. */
double double_factorial(int n);
/** @brief Binomial coefficient helper. */
double binomial(int m, int n);
///@}

/** @name Primitive Gaussian Integrals */
///@{
/** @brief Compute the Gaussian product center for two primitives. */
arma::rowvec gaussianCenter(const AtomicOrbital &aoA, const AtomicOrbital &aoB,
                            double aoA_exp, double aoB_exp);
/** @brief Single-term 1D Cartesian overlap contribution. */
double calc1DOverlap(double alpha, double beta, double X_A, double X_B,
                     double X_P, int l_A, int l_B, int i, int j);
/** @brief Full 1D overlap by summing valid Cartesian terms. */
double calcTotal1DOverlap(double alpha, double beta, double X_A, double X_B,
                          double X_P, int l_A, int l_B);
/** @brief Total 3D overlap between two primitive Gaussians. */
double calc3DOverlap(const AtomicOrbital &aoA, const PrimitiveGaussian &primA,
                     const AtomicOrbital &aoB, const PrimitiveGaussian &primB);
///@}

/** @name Contracted Integrals */
///@{
/** @brief Contracted AO overlap integral. */
double calcContractedOverlap(const AtomicOrbital &aoA, const AtomicOrbital &aoB);

/** @brief Build the molecule AO overlap matrix. */
arma::mat contractedOverlapMatrix(const Molecule &molecule);
///@}

#endif