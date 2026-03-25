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

    /* Load a system of atoms from a set of atom position and basis files */
    System System::from_files(std::string atoms_filepath, std::string basis_directory) {
        std::ifstream atoms_file(atoms_filepath);
        std::string line = "";

        // Validate basis_directory
        if (!fs::exists(basis_directory) || !fs::is_directory(basis_directory)) {
            std::cerr << "Invalid directory " << basis_directory << std::endl;
            throw std::runtime_error("Could not load basis files");
        }
        
        // Read in num elements from first line
        std::getline(atoms_file, line);
        std::stringstream sstream(line);
        int count;
        sstream >> count;
        std::vector<Atom> atoms;
        atoms.reserve(count);

        // Skip comment line
        std::getline(atoms_file, line);

        // Iterate through atoms and their positions
        while(std::getline(atoms_file, line)) {
            sstream = std::stringstream(line);
            short elm = 0;
            double x,y,z=0;
            std::cout << line << std::endl;
            sstream >> elm >> x >> y >> z;

            // Convert atomic number to symbol to try to find
            // a matching basis file by name.
            std::string atomic_symbol = Atom::get_number_symbol(elm);
            std::unordered_map<std::string, GaussianTemplate> orbital_templates{};

            for (const auto& entry : fs::directory_iterator(basis_directory)) {
                std::string name = entry.path().filename();
                std::string symbol = "";
                std::string orbital = "";
                u_short n_underscores = 0;

                // Extract element symbol and orbital from file name
                for (const auto& letter : name) {
                    if (letter == '_') {
                        n_underscores += 1;
                    }
                    else if (n_underscores == 0) {
                        symbol += letter;
                    }
                    else if (n_underscores == 1) {
                        orbital = letter;
                    }
                    else {
                        break;
                    }
                }
                
                // Extract template information
                if (symbol == atomic_symbol) {
                    json orbital_info = json::parse(
                        std::ifstream(entry.path())
                    );
                    std::vector<double> alphas{};
                    std::vector<double> weights{};
                    for(const auto& primitive: orbital_info["contracted_gaussians"]) {
                        alphas.emplace_back(primitive["exponent"]);
                        weights.emplace_back(primitive["contraction_coefficient"]);
                    }
                    orbital_templates.emplace(orbital, GaussianTemplate(alphas, weights));
                }
            }
            if (elm < 3) { // Hydrogen + Helium
                atoms.push_back(Atom({x,y,z}, elm, orbital_templates.at("s")));
            }
            else { // All other elements (in second row)
                atoms.push_back(Atom(
                    {x,y,z},
                    elm,
                    orbital_templates.at("s"),
                    orbital_templates.at("p")
                ));
            }
        }

        // Final, construct and return system
        return System(atoms);
    }
}