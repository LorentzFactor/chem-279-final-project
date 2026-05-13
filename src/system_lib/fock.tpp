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

  if (!sys.use_indo()) {
    for (const auto &atom : atoms) {
      for (const auto &orbital : atom.get_atomic_orbitals()) {
        double diag_term = 0;
        if (orbital.shell == 0) { // s shell
          diag_term -= atom.get_atom_constant("sI+A/2");
        } else if (orbital.shell == 1) { // p shell
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
  }

  if (sys.use_indo()) {
    int iorbital = 0;
    for (size_t iatom = 0; iatom < sys.num_atoms(); ++iatom) {
      const Atom &atom = sys.get_atom(iatom);
      double diagonal_interatomic_sum = 0;

      for (size_t jatom = 0; jatom < sys.num_atoms(); ++jatom) {
        if (jatom == iatom)  { continue; }
        const Atom &atom_j = sys.get_atom(jatom);

        diagonal_interatomic_sum +=
            (p_AA.at(jatom) - atom_j.get_atom_constant("Z_A")) *
            reduced_gamma(iatom, jatom);
      }

      int start_orbital = iorbital;
      for (size_t j = start_orbital; j < start_orbital + atom.num_orbitals(); ++j) {
        const GaussianContracted &mu = atom.get_atomic_orbital(j - start_orbital);
        for (size_t k = start_orbital; k < start_orbital + atom.num_orbitals(); ++k) {
          f_alpha(j, k) = 0;
          f_beta(j, k) = 0;

          if (k == j) {
            // First term fom Eq. 3.9 of INDO paper
            if (mu.shell == 0) {
              f_alpha(j, j) += atom.get_atom_constant("sINDO_U_MU_MU");
              f_beta(j, j) += atom.get_atom_constant("sINDO_U_MU_MU");
            } else {
              f_alpha(j, j) += atom.get_atom_constant("pINDO_U_MU_MU");
              f_beta(j, j) += atom.get_atom_constant("pINDO_U_MU_MU");
            }

            // double electron integral component from Eq. 3.9 of INDO paper
            for(size_t l = start_orbital; l < start_orbital + atom.num_orbitals(); ++l) {
              const GaussianContracted &lam = atom.get_atomic_orbital(l - start_orbital);
              f_alpha(j, j) += p_tot(l, l) * get_integral(mu, mu, lam, lam, atom)
                               - p_alpha(l, l) * get_integral(mu, lam, mu, lam, atom);
              f_beta(j, j) += p_tot(l, l) * get_integral(mu, mu, lam, lam, atom)
                                - p_beta(l, l) * get_integral(mu, lam, mu, lam, atom);
            }

            // Interatomic component from Eq. 3.9 of INDO paper
            f_alpha(j, k) += diagonal_interatomic_sum;
            f_beta(j, k) += diagonal_interatomic_sum;
            continue;
          }

          // Off-diagonal two-electron integral matrix elements from Eq. 3.9 of INDO paper
          const GaussianContracted &nu = atom.get_atomic_orbital(k - start_orbital);
          f_alpha(j, k) += (2.0*p_tot(j,k) - p_alpha(j,k)) * get_integral(mu, nu, mu, nu, atom)
                            - p_alpha(j,k) * get_integral(mu, mu, nu, nu, atom);
          f_beta(j, k) += (2.0*p_tot(j,k) - p_beta(j,k)) * get_integral(mu, nu, mu, nu, atom)
                            - p_beta(j,k) * get_integral(mu, mu, nu, nu, atom);
        }
        ++iorbital;
      }
    }
  }

  // Magnetic perturbation after CNDO2 diagonal is set (so diagonal Im parts are
  // kept)
  if constexpr (std::is_same_v<MatT, ComplexMat>) {
    const std::array<double, 3> gauge_origin = 
              molecule_center_of_mass(sys.atoms());
    const int dir = sys.get_field_dir();
    const RealMat L_k = sys.compute_angular_momentum_matrix(dir, gauge_origin);
    const std::complex<double> i_unit(0.0, 1.0);

    // dLambda / dBk = 0.5, apply that to so that lambda in terms of Bk
    const ComplexMat perturbation = 0.5 * i_unit * sys.get_lambda() * L_k;
    f_alpha += perturbation;
    f_beta += perturbation;
  }

  return {f_alpha, f_beta};
}
