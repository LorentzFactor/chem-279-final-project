#ifndef INTEGRATE_H
#define INTEGRATE_H

#include "orbitals.h"

/**
 * @brief Factorial n!.
 * @param n Non-negative integer.
 * @return n!.
 */
double factorial(int n);

/**
 * @brief Double factorial n!! (n * (n-2) * ... ).
 * @param n Integer (≤1 returns 1).
 * @return n!!.
 */
double double_factorial(int n);

/**
 * @brief Binomial coefficient C(m,n) = m! / (n! (m-n)!).
 * @param m Upper index.
 * @param n Lower index.
 * @return Binomial coefficient.
 */
double binomial(int m, int n);

/**
 * @brief Compute the Gaussian product center for two atomic orbitals.
 *
 * @param aoA First atomic orbital.
 * @param aoB Second atomic orbital.
 * @param aoA_exp Exponent of the primitive on `aoA`.
 * @param aoB_exp Exponent of the primitive on `aoB`.
 * @return Cartesian coordinates of the product center.
 */
arma::rowvec gaussianCenter(const AtomicOrbital &aoA, const AtomicOrbital &aoB,
                            double aoA_exp, double aoB_exp);

/**
 * @brief Compute the 1D overlap contribution for two Cartesian Gaussian
 * primitives.
 *
 * @param alpha Exponent of the first primitive.
 * @param beta Exponent of the second primitive.
 * @param X_A Coordinate of the first center along the current axis.
 * @param X_B Coordinate of the second center along the current axis.
 * @param X_P Product center coordinate along the current axis.
 * @param l_A Angular momentum of the first primitive along the axis.
 * @param l_B Angular momentum of the second primitive along the axis.
 * @param i Summation index for the first primitive.
 * @param j Summation index for the second primitive.
 * @return 1D overlap integral value along the chosen axis.
 */
double calc1DOverlap(double alpha, double beta, double X_A, double X_B,
                     double X_P, int l_A, int l_B, int i, int j);

double calcTotal1DOverlap(double alpha, double beta, double X_A, double X_B,
                          double X_P, int l_A, int l_B);

/**
 * @brief Compute the 3D overlap integral between two primitive Gaussians.
 *
 * @param aoA Atomic orbital associated with the first primitive.
 * @param primA First primitive Gaussian.
 * @param aoB Atomic orbital associated with the second primitive.
 * @param primB Second primitive Gaussian.
 * @return Total 3D overlap between the two primitives.
 */
double calc3DOverlap(const AtomicOrbital &aoA, const PrimitiveGaussian &primA,
                     const AtomicOrbital &aoB, const PrimitiveGaussian &primB);

/**
 * @brief Compute the contracted overlap between two atomic orbitals.
 *
 * @param aoA First contracted atomic orbital.
 * @param aoB Second contracted atomic orbital.
 * @return Contracted overlap integral between `aoA` and `aoB`.
 */
double calcContractedOverlap(AtomicOrbital &aoA, AtomicOrbital &aoB);

/**
 * @brief Build the contracted overlap matrix for a molecule.
 *
 * @param molecule Molecule providing atomic orbitals.
 * @return Overlap matrix \(S\).
 */
arma::mat contractedOverlapMatrix(const Molecule &molecule);

#endif