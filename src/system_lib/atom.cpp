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
            orbital_names_.emplace_back("p");
        }
    }

    /* Convert an atomic number to its symbol */
    std::string Atom::get_number_symbol(const short& atomic_number) {
        return symbol_list_[atomic_number-1];
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
        else {
            throw std::runtime_error("Constant name not found");
        }
    }
}