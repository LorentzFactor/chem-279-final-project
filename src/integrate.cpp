#include "orbitals.h"
#define _USE_MATH_DEFINES
#include <armadillo>
#include <cmath>
#include <stdexcept>

#include "integrate.h"

/**
 * @brief Factorial n!.
 * @param n Non-negative integer.
 * @return n!.
 */
double factorial(int n) {
  if (n < 0) {
    throw std::runtime_error("n must not be less than zero (factorial).");
  } else if (n <= 1) {
    return 1.0;
  }
  return n * factorial(n - 1);
}

/**
 * @brief Double factorial n!! (n * (n-2) * ... ).
 * @param n Integer (≤1 returns 1).
 * @return n!!.
 */
double double_factorial(int n) {
  if (n <= 1) {
    return 1.0;
  }
  return n * double_factorial(n - 2);
}

/**
 * @brief Binomial coefficient C(m,n) = m! / (n! (m-n)!).
 * @param m Upper index.
 * @param n Lower index.
 * @return Binomial coefficient.
 */
double binomial(int m, int n) {
  return factorial(m) / (factorial(n) * factorial((m - n)));
}

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
                            double aoA_exp, double aoB_exp) {
  double alpha = aoA_exp;
  double beta = aoB_exp;

  // determine product center for each dimension
  arma::rowvec center(3);
  for (arma::uword xyz = 0; xyz < 3; ++xyz) {
    center(xyz) =
        ((alpha * aoA.coords(xyz)) + (beta * aoB.coords(xyz))) / (alpha + beta);
  }
  return center;
}

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
                     double X_P, int l_A, int l_B, int i, int j) {

  // Decompose analytical formula into smaller terms
  double front_term =
      std::exp(-(alpha * beta * ((X_A - X_B) * (X_A - X_B))) / (alpha + beta));
  double pi_term = std::sqrt(M_PI / (alpha + beta));
  double binomial_term = binomial(l_A, i) * binomial(l_B, j);
  double end_num_term = double_factorial(i + j - 1) *
                        std::pow((X_P - X_A), (l_A - i)) *
                        std::pow((X_P - X_B), (l_B - j));
  double end_den_term = std::pow(2 * (alpha + beta), (i + j) / 2);

  // Recombine terms for final result
  return front_term * pi_term * binomial_term * (end_num_term / end_den_term);
}

double calcTotal1DOverlap(double alpha, double beta, double X_A, double X_B,
                          double X_P, int l_A, int l_B) {
  if (l_A < 0 || l_B < 0) {
    return 0.0;
  }

  double overlap = 0.0;
  for (int i = 0; i < l_A + 1; ++i) {
    for (int j = 0; j < l_B + 1; ++j) {
      if ((i + j) % 2 != 0)
        continue;
      overlap += calc1DOverlap(alpha, beta, X_A, X_B, X_P, l_A, l_B, i, j);
    }
  }
  return overlap;
}

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
                     const AtomicOrbital &aoB, const PrimitiveGaussian &primB) {
  // Initialize overlap vec and calculate overlap in each dimension
  double total_overlap;
  arma::rowvec gauss_overlap_xyz(3, arma::fill::zeros);
  arma::rowvec center =
      gaussianCenter(aoA, aoB, primA.exponent, primB.exponent);
  int num_dims = aoA.momentum.n_cols;
  for (int xyz = 0; xyz < num_dims; ++xyz) {
    gauss_overlap_xyz.at(xyz) = calcTotal1DOverlap(
        primA.exponent, primB.exponent, aoA.coords(xyz), aoB.coords(xyz),
        center(xyz), aoA.momentum(xyz), aoB.momentum(xyz));
  }
  // Multiple gaussians across all dimensions, store to overlap matrix
  total_overlap = arma::prod(gauss_overlap_xyz);
  return total_overlap;
}

/**
 * @brief Compute the contracted overlap between two atomic orbitals.
 *
 * @param aoA First contracted atomic orbital.
 * @param aoB Second contracted atomic orbital.
 * @return Contracted overlap integral between `aoA` and `aoB`.
 */
double calcContractedOverlap(const AtomicOrbital &aoA,
                             const AtomicOrbital &aoB) {
  double total_overlap = 0.0;
  // nested loop to iterate through all the primitive gaussians in each basis
  // func
  for (int a_idx = 0; a_idx < aoA.prim_gauss.size(); ++a_idx) {
    for (int b_idx = 0; b_idx < aoB.prim_gauss.size(); ++b_idx) {
      double overlap = calc3DOverlap(aoA, aoA.prim_gauss.at(a_idx), aoB,
                                     aoB.prim_gauss.at(b_idx));
      double contract_product = aoA.prim_gauss.at(a_idx).contraction *
                                aoB.prim_gauss.at(b_idx).contraction;
      double normal_product =
          aoA.prim_gauss.at(a_idx).norm * aoB.prim_gauss.at(b_idx).norm;
      total_overlap += contract_product * normal_product * overlap;
    }
  }

  return total_overlap;
}

/**
 * @brief Build the contracted overlap matrix for a molecule.
 *
 * @param molecule Molecule providing atomic orbitals.
 * @return Overlap matrix \(S\).
 */
arma::mat contractedOverlapMatrix(const Molecule &molecule) {
  int N = molecule.getN();
  arma::mat S(N, N, arma::fill::zeros);

  // Determine overlap of all Shell A and B function pairs
  for (arma::uword a_idx = 0; a_idx < N; ++a_idx) {
    for (arma::uword b_idx = 0; b_idx < N; ++b_idx) {
      S(a_idx, b_idx) = calcContractedOverlap(molecule.getAtomicOrbital(a_idx),
                                              molecule.getAtomicOrbital(b_idx));
    }
  }
  return S;
}
