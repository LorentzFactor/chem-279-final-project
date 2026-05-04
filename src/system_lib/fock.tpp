#include <array>
#include <complex>
#include <stdexcept>
#include <type_traits>

template <class MatT, class SystemT>
std::pair<MatT, MatT> build_cndo2_fock(SystemT &sys, const MatT &p_alpha,
                                       const MatT &p_beta) {
  CNDO2RealIntegrals &I = sys.real_integrals();
  const std::vector<Atom> &atoms = sys.atoms();

  RealMat &gamma_cache = I.gamma_;
  RealMat &gamma_reduced_cache = I.gamma_reduced_;
  RealMat &beta_cache = I.beta_;

  RealMat gamma = compute_gamma_matrix(gamma_cache, sys, atoms);
  RealMat reduced_gamma =
      compute_reduced_gamma_matrix(gamma_reduced_cache, sys);
  RealMat S = sys.compute_overlap_matrix();
  RealMat beta = compute_beta_matrix(beta_cache, sys, atoms);

  const MatT SxB = arma::conv_to<MatT>::from(S % beta);
  const MatT G = arma::conv_to<MatT>::from(gamma);
  MatT f_alpha = SxB - p_alpha % G;
  MatT f_beta = SxB - p_beta % G;

  MatT p_tot = p_alpha + p_beta;

  RealVec p_AA = arma::zeros(sys.num_atoms());
  RealVec p_diag = arma::real(p_tot.diag());
  for (size_t iatom = 0; iatom < sys.num_atoms(); ++iatom) {
    std::array<size_t, 2> indices = sys.get_atom_orbital_idxs()[iatom];
    p_AA(iatom) = arma::sum(p_diag.subvec(indices[0], indices[1] - 1));
  }

  int iorbital = 0;
  int iatom = 0;
  for (const auto &atom : atoms) {
    for (const auto &orbital : atom.get_atomic_orbitals()) {
      double diag_term = 0;

      if (orbital.shell == 0) {
        diag_term -= atom.get_atom_constant("sI+A/2");
      } else if (orbital.shell == 1) {
        diag_term -= atom.get_atom_constant("pI+A/2");
      } else {
        throw std::runtime_error("unsupported shell");
      }

      for (size_t jatom = 0; jatom < sys.num_atoms(); ++jatom) {
        if ((int)jatom == iatom) {
          continue;
        }
        const Atom &atom_c = sys.get_atom(jatom);
        double gamma_AC = reduced_gamma(iatom, jatom);
        diag_term +=
            (p_AA.at(jatom) - atom_c.get_atom_constant("Z_A")) * gamma_AC;
      }

      double diag_term_a =
          diag_term + ((p_AA.at(iatom) - atom.get_atom_constant("Z_A")) -
                       (std::real(p_alpha(iorbital, iorbital)) - 0.5)) *
                          gamma(iorbital, iorbital);
      double diag_term_b =
          diag_term + ((p_AA.at(iatom) - atom.get_atom_constant("Z_A")) -
                       (std::real(p_beta(iorbital, iorbital)) - 0.5)) *
                          gamma(iorbital, iorbital);

      f_alpha(iorbital, iorbital) = diag_term_a;
      f_beta(iorbital, iorbital) = diag_term_b;
      ++iorbital;
    }
    ++iatom;
  }

  // Magnetic perturbation after CNDO2 diagonal is set (so diagonal Im parts are kept)
  if constexpr (std::is_same_v<MatT, ComplexMat>) {
    const std::array<double, 3> gauge_origin =
        molecule_center_of_mass(sys.atoms());
    const int dir = sys.get_field_dir();
    const RealMat L_k =
        sys.compute_angular_momentum_matrix(dir, gauge_origin);
    const std::complex<double> i_unit(0.0, 1.0);
    const ComplexMat perturbation = i_unit * sys.get_lambda() * L_k;
    f_alpha += perturbation;
    f_beta += perturbation;
  }

  return {f_alpha, f_beta};
}
