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
        public:
            Atom(std::array<double,3> position, short atomic_number, GaussianTemplate s_basis);
            Atom(std::array<double,3> position, short atomic_number, GaussianTemplate s_basis, GaussianTemplate p_basis);
            static std::string get_number_symbol(const short& atomic_number);
            std::string get_symbol() const {return atomic_symbol_;};
    };

    struct System {
        private:
            std::vector<Atom> atoms_;
        public:
            System(std::vector<Atom> atoms): atoms_(atoms){};
            static System from_files(std::string atoms_filepath, std::string basis_directory);
    };
}
