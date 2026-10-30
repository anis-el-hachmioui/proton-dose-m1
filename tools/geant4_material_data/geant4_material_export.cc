#include "material_definition.hh"

#include "G4RunManagerFactory.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4EmCalculator.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4HadronicProcessStore.hh"
#include "G4SystemOfUnits.hh"

#include "QGSP_BIC_HP.hh"

#include "G4VUserDetectorConstruction.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4ThreeVector.hh"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// Extraction grid
// -----------------------------------------------------------------------------
// Logarithmic sampling resolves the rapid low-energy variation while retaining
// coverage of the full clinical proton-energy range.
constexpr double ENERGY_MIN_MEV = 1.0e-4;
constexpr double ENERGY_MAX_MEV = 300.0;
constexpr int    N_ENERGY        = 6000;


// -----------------------------------------------------------------------------
// Minimal Geant4 application
// -----------------------------------------------------------------------------

class DummyDetector final : public G4VUserDetectorConstruction {
public:
    explicit DummyDetector(
        MaterialDefinition material_definition
    )
        : material_definition_(
            std::move(material_definition)
        ) {}

    G4VPhysicalVolume* Construct() override {
        auto* nist = G4NistManager::Instance();

        auto* world_material = nist->FindOrBuildMaterial("G4_AIR");
        auto* target_material =
            build_material(
                material_definition_
            );

        if (world_material == nullptr || target_material == nullptr) {
            throw std::runtime_error(
                "Could not construct Geant4 material '" + material_definition_.name + "'."
            );
        }

        auto* solid_world = new G4Box(
            "World",
            50.0 * cm,
            50.0 * cm,
            50.0 * cm
        );

        auto* logical_world = new G4LogicalVolume(
            solid_world,
            world_material,
            "World"
        );

        auto* physical_world = new G4PVPlacement(
            nullptr,
            G4ThreeVector(),
            logical_world,
            "World",
            nullptr,
            false,
            0,
            false
        );

        auto* solid_target = new G4Box(
            "Target",
            10.0 * cm,
            10.0 * cm,
            10.0 * cm
        );

        auto* logical_target = new G4LogicalVolume(
            solid_target,
            target_material,
            "Target"
        );

        new G4PVPlacement(
            nullptr,
            G4ThreeVector(),
            logical_target,
            "Target",
            logical_world,
            false,
            0,
            false
        );

        return physical_world;
    }

private:
    MaterialDefinition material_definition_;
};


class DummyPrimary final : public G4VUserPrimaryGeneratorAction {
public:
    void GeneratePrimaries(G4Event*) override {}
};


class DummyActions final : public G4VUserActionInitialization {
public:
    void Build() const override {
        SetUserAction(new DummyPrimary());
    }
};


// TOPAS currently exposes QGSP_BIC_HP as its standard hadronic family.
// We use that reference physics list for hadronic process construction, but
// replace its EM constructor with option4 so S(E) is evaluated with the same
// EM family used in the deterministic/TOPAS matched-physics workflow.
class ExtractionPhysicsList final : public QGSP_BIC_HP {
public:
    ExtractionPhysicsList()
        : QGSP_BIC_HP(0) {
        SetVerboseLevel(0);
        ReplacePhysics(new G4EmStandardPhysics_option4(0));
    }
};


// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

std::string safe_filename(const std::string& material) {
    std::string result = material;
    for (char& c : result) {
        if (c == '/' || c == '\\' || c == ' ') {
            c = '_';
        }
    }
    return result;
}


std::vector<double> make_energy_grid_mev() {
    std::vector<double> energies;
    energies.reserve(N_ENERGY);

    const double log_min = std::log(ENERGY_MIN_MEV);
    const double log_max = std::log(ENERGY_MAX_MEV);

    for (int i = 0; i < N_ENERGY; ++i) {
        const double t =
            static_cast<double>(i) /
            static_cast<double>(N_ENERGY - 1);

        energies.push_back(
            std::exp(log_min + t * (log_max - log_min))
        );
    }

    return energies;
}


void replace_file(const fs::path& temporary, const fs::path& destination) {
    std::error_code ec;

    if (fs::exists(destination)) {
        fs::remove(destination, ec);
        if (ec) {
            throw std::runtime_error(
                "Could not remove old file '" +
                destination.string() +
                "': " +
                ec.message()
            );
        }
    }

    fs::rename(temporary, destination, ec);
    if (ec) {
        throw std::runtime_error(
            "Could not rename '" +
            temporary.string() +
            "' to '" +
            destination.string() +
            "': " +
            ec.message()
        );
    }
}


void write_scalar_material_row(
    const fs::path& csv_path,
    const std::string& material,
    double density_kg_m3,
    double radiation_length_m
) {
    std::vector<std::string> retained_rows;

    if (fs::exists(csv_path)) {
        std::ifstream old_file(csv_path);

        if (!old_file) {
            throw std::runtime_error(
                "Could not read existing scalar material file: " +
                csv_path.string()
            );
        }

        std::string line;
        bool first_line = true;

        while (std::getline(old_file, line)) {
            if (first_line) {
                first_line = false;
                continue;
            }

            if (line.empty()) {
                continue;
            }

            const auto comma = line.find(',');
            const std::string old_material =
                comma == std::string::npos
                    ? line
                    : line.substr(0, comma);

            if (old_material != material) {
                retained_rows.push_back(line);
            }
        }
    }

    const fs::path temporary = csv_path.string() + ".tmp";

    std::ofstream out(temporary);
    if (!out) {
        throw std::runtime_error(
            "Could not write scalar material file: " +
            temporary.string()
        );
    }

    out << "material,density_kg_m3,radiation_length_m\n";
    for (const auto& row : retained_rows) {
        out << row << '\n';
    }

    out << std::setprecision(17)
        << material << ','
        << density_kg_m3 << ','
        << radiation_length_m << '\n';

    out.close();
    replace_file(temporary, csv_path);
}


template <typename Evaluator>
void write_energy_table(
    const fs::path& csv_path,
    const std::string& value_header,
    const std::vector<double>& energies_mev,
    Evaluator evaluator
) {
    const fs::path temporary = csv_path.string() + ".tmp";

    std::ofstream out(temporary);
    if (!out) {
        throw std::runtime_error(
            "Could not write energy table: " +
            temporary.string()
        );
    }

    out << "energy_mev," << value_header << '\n';
    out << std::setprecision(17);

    for (const double energy_mev : energies_mev) {
        const double value = evaluator(energy_mev);

        if (!std::isfinite(value) || value < 0.0) {
            std::ostringstream message;
            message
                << "Invalid value " << value
                << " at E=" << energy_mev
                << " MeV while writing "
                << csv_path.string();

            throw std::runtime_error(message.str());
        }

        out << energy_mev << ',' << value << '\n';
    }

    out.close();
    replace_file(temporary, csv_path);
}


// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::string material_name = "<unknown>";

    try {
        const auto material_definition =
            parse_material_definition(
                argc,
                argv
            );

        material_name =
            material_definition.name;

        const std::string file_name =
            safe_filename(
                material_name
            );

        // Python deliberately runs this executable with cwd=repository root.
        const fs::path repo_root = fs::current_path();
        const fs::path data_dir = repo_root / "data";
        const fs::path stopping_dir = data_dir / "stopping_power";
        const fs::path attenuation_dir = data_dir / "nuclear_attenuation";

        fs::create_directories(stopping_dir);
        fs::create_directories(attenuation_dir);

        auto* run_manager =
            G4RunManagerFactory::CreateRunManager(
                G4RunManagerType::Serial
            );

        run_manager->SetUserInitialization(
            new DummyDetector(
                material_definition
            )
        );

        run_manager->SetUserInitialization(
            new ExtractionPhysicsList()
        );

        run_manager->SetUserInitialization(
            new DummyActions()
        );

        run_manager->Initialize();
        run_manager->RunInitialization();

        auto* proton =
            G4ParticleTable::GetParticleTable()
                ->FindParticle("proton");

        auto* material =
            build_material(
                material_definition
            );

        if (proton == nullptr) {
            throw std::runtime_error(
                "Could not obtain the Geant4 proton definition."
            );
        }

        if (material == nullptr) {
            throw std::runtime_error(
                "Could not obtain Geant4 material '" +
                material_name +
                "'."
            );
        }

        // -----------------------------------------------------------------
        // Energy-independent material properties
        // -----------------------------------------------------------------
        const double density_kg_m3 =
            material->GetDensity() / (kg / m3);

        const double radiation_length_m =
            material->GetRadlen() / m;

        if (
            !std::isfinite(density_kg_m3) ||
            density_kg_m3 <= 0.0
        ) {
            throw std::runtime_error(
                "Geant4 returned an invalid material density."
            );
        }

        if (
            !std::isfinite(radiation_length_m) ||
            radiation_length_m <= 0.0
        ) {
            throw std::runtime_error(
                "Geant4 returned an invalid radiation length."
            );
        }

        write_scalar_material_row(
            data_dir / "materials.csv",
            material_name,
            density_kg_m3,
            radiation_length_m
        );

        // -----------------------------------------------------------------
        // Energy-dependent quantities
        // -----------------------------------------------------------------
        const auto energies_mev =
            make_energy_grid_mev();

        G4EmCalculator em_calculator;
        em_calculator.SetVerbose(0);

        // CSDA stopping power used by the deterministic model.
        //
        // We intentionally use ELECTRONIC dE/dx here rather than adding a
        // separate nuclear-stopping term to S(E). Nuclear reactions are
        // represented separately through Sigma_r below.
        write_energy_table(
            stopping_dir /
                (file_name + ".csv"),
            "electronic_stopping_power_mev_per_m",
            energies_mev,
            [&](double energy_mev) {
                const G4double dedx =
                    em_calculator.ComputeElectronicDEDX(
                        energy_mev * MeV,
                        proton,
                        material
                    );

                return dedx / (MeV / m);
            }
        );

        // Primary-proton nuclear removal coefficient.
        //
        // Definition used in this thesis pipeline:
        //
        //   Sigma_r(E) =
        //       Geant4 proton INELASTIC macroscopic cross section
        //       in the selected material.
        //
        // This treats a proton undergoing a nuclear inelastic interaction as
        // removed from the primary-proton population. It does not model the
        // dose contribution of nuclear secondaries.
        auto* hadronic_store =
            G4HadronicProcessStore::Instance();

        hadronic_store->SetVerbose(0);

        write_energy_table(
            attenuation_dir /
                (file_name + ".csv"),
            "nuclear_removal_coefficient_per_m",
            energies_mev,
            [&](double energy_mev) {
                const G4double sigma_r =
                    hadronic_store
                        ->GetInelasticCrossSectionPerVolume(
                            proton,
                            energy_mev * MeV,
                            material
                        );

                return sigma_r / (1.0 / m);
            }
        );

        run_manager->RunTermination();
        delete run_manager;

        std::cout
            << "Extracted " << material_name << '\n'
            << "  density            = "
            << std::setprecision(12)
            << density_kg_m3
            << " kg/m^3\n"
            << "  radiation length   = "
            << radiation_length_m
            << " m\n"
            << "  stopping power     = "
            << (
                stopping_dir /
                (file_name + ".csv")
            ).string()
            << '\n'
            << "  nuclear attenuation= "
            << (
                attenuation_dir /
                (file_name + ".csv")
            ).string()
            << '\n';

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "Material extraction failed for '"
            << material_name
            << "': "
            << error.what()
            << '\n';

        return 1;
    }
}
