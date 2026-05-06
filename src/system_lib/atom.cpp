#include "system_lib.h"

using namespace gaussian_lib;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace system_lib {
    Atom::Atom(std::array<double,3> position, short atomic_number, GaussianTemplate s_basis)
        : position_(position), atomic_number_(atomic_number)
    {
        if(atomic_number_ > 2) {
            throw std::runtime_error("A p orbital basis must be specified for atoms above helium.");
        }
        atomic_symbol_ = symbol_list_[atomic_number-1];

        // Add s orbital at location of the atom
        atomic_orbitals_ = { s_basis.toFunction(position, {0,0,0}) };
        orbital_names_ = { "1s" };

        F0 = calculate_gamma(atomic_orbitals_.at(0), atomic_orbitals_.at(0));
    }

    Atom::Atom(std::array<double,3> position, short atomic_number, GaussianTemplate s_basis, GaussianTemplate p_basis)
        : position_(position), atomic_number_(atomic_number)
    {
        if(atomic_number_ < 3) {
            throw std::runtime_error("A p orbital basis should not be specified for atoms below lithium.");
        }
        atomic_symbol_ = symbol_list_[atomic_number-1];

        atomic_orbitals_.reserve(4);
        orbital_names_.reserve(4);

        // S orbital
        atomic_orbitals_ = {s_basis.toFunction(position, {0,0,0})};
        orbital_names_.emplace_back("s");

        // P orbitals
        for (size_t idim = 0; idim < 3; idim++) {
            std::array<char,3> momentums = {0,0,0};
            momentums[idim] = 1;
            atomic_orbitals_.push_back(p_basis.toFunction(position, momentums));
            std::string p_label = "p";
            p_label += (idim == 0) ? "x" : (idim == 1) ? "y" : "z";
            orbital_names_.emplace_back(p_label);
        }

        F0 = calculate_gamma(atomic_orbitals_.at(0), atomic_orbitals_.at(0));
    }

    /* Convert an atomic number to its symbol */
    const std::string& Atom::get_number_symbol(const short& atomic_number) {
        if (atomic_number < 1 ||
            atomic_number > static_cast<short>(symbol_list_.size())) {
            throw std::runtime_error("Atomic number out of supported range: " +
                                     std::to_string(atomic_number));
        }
        return symbol_list_[static_cast<size_t>(atomic_number - 1)];
    }

    /* Convert an atomic symbol to its number */
    const short Atom::get_symbol_number(const std::string& atomic_symbol) {
        auto it = std::find(symbol_list_.begin(), symbol_list_.end(), atomic_symbol);
        if (it != symbol_list_.end()) {
            return std::distance(symbol_list_.begin(), it) + 1;
        }
        else {
            throw std::runtime_error("Atomic symbol not found: " + atomic_symbol);
        }
    }

    double Atom::get_atom_constant(const std::string& const_name) const {
        if (const_name == "sI+A/2") {
            return sI_plus_A_.at(get_symbol());
        }
        else if (const_name == "pI+A/2") {
            return pI_plus_A_.at(get_symbol());
        }
        else if(const_name == "neg_beta") {
            return neg_beta_.at(get_symbol());
        }
        else if (const_name == "Z_A") {
            return Z_A.at(get_symbol());
        }
        else if (const_name == "F0") { // i.e. gamma_AA
            return F0;
        }
        else if (const_name == "F2") {
            return F2_.at(get_symbol());
        }
        else if (const_name == "G1") {
            return G1_.at(get_symbol());
        }
        else if (const_name == "sINDO_U_MU_MU") {
            return sINDO_U_MU_MU_.at(get_symbol());
        }
        else if (const_name == "pINDO_U_MU_MU") {
            return pINDO_U_MU_MU_.at(get_symbol());
        }
        else {
            throw std::runtime_error("Constant name not found");
        }
    }

    /* Compute the distance between atom a1 and atom a2. */
    double distance(const Atom& a1, const Atom& a2) {
        auto& ra = a1.get_atomic_orbitals().at(0).center;
        auto& rb = a2.get_atomic_orbitals().at(0).center;
        arma::vec3 RA = arma::vec3(ra.data());
        arma::vec3 RB = arma::vec3(rb.data());
        return arma::norm(RA-RB);
    }
}