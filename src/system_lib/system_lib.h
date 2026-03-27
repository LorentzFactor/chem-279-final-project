#pragma once

#include <filesystem>
#include<iostream>
#include <fstream>

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
        private:
            std::vector<Atom> atoms_;
            size_t num_orbitals_;
            std::vector<std::array<size_t, 2>> atom_orbital_idxs;
        public:
            System(const std::vector<Atom>& atoms);
            static System from_files(std::string atoms_filepath, std::string basis_directory);
            arma::mat compute_overlap_matrix() const;
            arma::mat compute_gamma_matrix() const;
            arma::mat compute_reduced_gamma_matrix() const;
            arma::mat compute_beta_matrix() const;
            double compute_nuclear_energy() const;
            std::pair<arma::mat,arma::mat> compute_cndo_f_matrix(const arma::mat& p_alpha, const arma::mat& p_beta) const;
            std::pair<arma::mat,arma::mat> compute_h_core() const;
            size_t num_orbitals() const;
    };

    double distance(const Atom& a1, const Atom& a2);
}
