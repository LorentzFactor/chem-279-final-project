#pragma once

#include <filesystem>
#include<iostream>
#include <fstream>
#include <memory>

#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <armadillo>
#include <nlohmann/json.hpp> 
#include "gaussian_lib/gaussian_lib.h"

using namespace gaussian_lib;

namespace system_lib {
    struct Atom {
        private:
            std::array<double,3> position_; // Position of atom
            short atomic_number_; // Its atomic number
            std::string atomic_symbol_; // Atomic symbol
            inline static const std::array<std::string, 9> symbol_list_ { // Ordered list of atomic symbols
                "H", "He", "Li", "Be", "B", "C", "N", "O", "F"
            };
            std::vector<GaussianContracted> atomic_orbitals_; // A vector of the atom's orbitals
            std::vector<std::string> orbital_names_; // A vector of the same length as orbitals with their names
            inline static const std::unordered_map<std::string, double> sI_plus_A_ {
                {"H", 7.176},
                {"C", 14.051},
                {"N", 19.361},
                {"O", 25.390},
                {"F", 32.272},
            };
            inline static const std::unordered_map<std::string, double> pI_plus_A_ {
                {"C", 5.572},
                {"N", 7.275},
                {"O", 9.111},
                {"F", 11.080}
            };
            inline static const std::unordered_map<std::string, double> neg_beta_ {
                {"H", -9},
                {"C", -21},
                {"N", -25},
                {"O", -31},
                {"F", -39},
            };
            inline static const std::unordered_map<std::string, double> Z_A {
                {"H", 1},
                {"C", 4},
                {"N", 5},
                {"O", 6},
                {"F", 7},
            };
        public:
            Atom(std::array<double,3> position, short atomic_number, GaussianTemplate s_basis);
            Atom(std::array<double,3> position, short atomic_number, GaussianTemplate s_basis, GaussianTemplate p_basis);
            static const std::string& get_number_symbol(const short& atomic_number);
            double get_atom_constant(const std::string& const_name) const;
            std::string get_symbol() const {return atomic_symbol_;};
            const std::vector<GaussianContracted>& get_atomic_orbitals() const {return atomic_orbitals_;};
            int num_orbitals() const {return atomic_orbitals_.size();};
            const std::vector<std::string>& get_orbital_names() const {return orbital_names_;};
    };

    struct System {
        protected:
            std::vector<Atom> atoms_;
            size_t num_orbitals_;
            std::vector<std::array<size_t, 2>> atom_orbital_idxs;
            static std::vector<Atom> atoms_from_files_(std::string atoms_filepath, std::string basis_directory);
        public:
            System(const std::vector<Atom>& atoms);
            arma::mat compute_overlap_matrix() const;
            arma::cube compute_overlap_matrix_gradient() const;
            size_t num_orbitals() const;
            size_t num_atoms() const;
            virtual double compute_electronic_energy() const = 0;
            double compute_nuclear_energy() const;
            double compute_total_energy() const;
    };

    struct CNDO2System : public System {
        private:
            // F matrix + energy states
            arma::mat p_alpha_;
            arma::mat p_beta_;
            arma::mat f_alpha_;
            arma::mat f_beta_;
            arma::mat mos_alpha_;
            arma::mat mos_beta_;
            arma::vec E_alpha_;
            arma::vec E_beta_;

            // Electron counts
            double p_;
            double q_;

            // Gradients
            arma::cube S_uv_R_;

        protected:
            std::pair<arma::mat,arma::mat> compute_cndo_f_matrix_internal(
                const arma::mat& p_alpha,
                const arma::mat& p_beta
            ) const;

        public:
            CNDO2System(const std::vector<Atom>& atoms, int p, int q);
            static CNDO2System from_files(std::string atoms_filepath, std::string basis_directory, int p, int q);
            
            void set_p(const arma::mat& new_p_alpha, const arma::mat& new_p_beta);
            const arma::mat& get_p_alpha() const {return p_alpha_;};
            const arma::mat& get_p_beta() const {return p_beta_;};

            const arma::mat& get_f_alpha() const {return f_alpha_;};
            const arma::mat& get_f_beta() const {return f_beta_;};

            void set_nelectrons(int p, int q) {p_ = (double) p; q_ = (double) q;};
            int get_nalpha() const {return (int) p_;};
            int get_nbeta() const {return (int) q_;};

            const arma::mat& get_MOs_alpha() const {return mos_alpha_;};
            const arma::mat& get_MOs_beta() const {return mos_beta_;};
            arma::mat get_occupied_MOs_alpha() const;
            arma::mat get_occupied_MOs_beta() const;

            arma::rowvec get_E_alpha() const {return E_alpha_.as_row();};
            arma::rowvec get_E_beta() const {return E_beta_.as_row();};

            double get_electron_density(const std::array<double,3>& position) const;
            arma::cube get_electron_density_3d_grid(
                const std::array<double,2>& x_range,
                const std::array<double,2>& y_range,
                const std::array<double,2>& z_range,
                int nsamples_per_side
            ) const;
            
            arma::mat compute_gamma_matrix() const;
            arma::mat compute_reduced_gamma_matrix() const;
            arma::mat compute_beta_matrix() const;
            std::pair<arma::mat,arma::mat> compute_cndo_f_matrix() const;
            arma::mat compute_h_core() const;
            double compute_electronic_energy() const override;
    };

    double distance(const Atom& a1, const Atom& a2);
}
