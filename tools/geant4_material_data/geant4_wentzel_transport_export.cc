#include "material_definition.hh"

#include "G4RunManagerFactory.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessTable.hh"
#include "G4ProcessVector.hh"
#include "G4VProcess.hh"
#include "G4VMultipleScattering.hh"
#include "G4VMscModel.hh"
#include "G4WentzelVIModel.hh"
#include "G4EmCalculator.hh"
#include "G4EmParameters.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4ProductionCuts.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4Version.hh"

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

namespace {

constexpr double ENERGY_MIN_MEV = 1.0e-4;
constexpr double ENERGY_MAX_MEV = 300.0;
constexpr int N_ENERGY = 6000;
constexpr double TOPAS_PRODUCTION_CUT_M = 100.0;


class DummyDetector final : public G4VUserDetectorConstruction {
public:
    explicit DummyDetector(
        MaterialDefinition material_definition
    )
        : material_definition_(std::move(material_definition)) {}

    G4VPhysicalVolume* Construct() override {
        auto* nist = G4NistManager::Instance();
        auto* world_material = nist->FindOrBuildMaterial("G4_AIR");
        auto* target_material = build_material(material_definition_);

        if (world_material == nullptr || target_material == nullptr) {
            throw std::runtime_error(
                "Could not construct Geant4 material '" +
                material_definition_.name +
                "'."
            );
        }

        auto* world_solid = new G4Box("World", 50.0 * cm, 50.0 * cm, 50.0 * cm);
        auto* world_logical = new G4LogicalVolume(
            world_solid,
            world_material,
            "World"
        );
        auto* world_physical = new G4PVPlacement(
            nullptr,
            G4ThreeVector(),
            world_logical,
            "World",
            nullptr,
            false,
            0,
            false
        );

        auto* target_solid = new G4Box("Target", 10.0 * cm, 10.0 * cm, 10.0 * cm);
        auto* target_logical = new G4LogicalVolume(
            target_solid,
            target_material,
            "Target"
        );
        new G4PVPlacement(
            nullptr,
            G4ThreeVector(),
            target_logical,
            "Target",
            world_logical,
            false,
            0,
            false
        );

        return world_physical;
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


// Match the validated TOPAS EM configuration. TOPAS applies the two EM
// parameter commands before initialization, then inactivates CoulombScat at
// the sequence stage after the WentzelVI transport table has been built.
class ExtractionPhysicsList final : public QGSP_BIC_HP {
public:
    ExtractionPhysicsList()
        : QGSP_BIC_HP(0) {
        SetVerboseLevel(0);
        ReplacePhysics(new G4EmStandardPhysics_option4(0));

        auto* parameters = G4EmParameters::Instance();
        parameters->SetLossFluctuations(false);
        parameters->SetMuHadLateralDisplacement(false);
    }

    void SetCuts() override {
        // Establish Geant4 defaults for any unmentioned particle, then apply
        // the three explicit TOPAS cuts relevant to the validated setup.
        SetCutsWithDefault();
        SetCutValue(TOPAS_PRODUCTION_CUT_M * m, "gamma");
        SetCutValue(TOPAS_PRODUCTION_CUT_M * m, "e-");
        SetCutValue(TOPAS_PRODUCTION_CUT_M * m, "e+");
    }
};


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
        const double fraction =
            static_cast<double>(i) / static_cast<double>(N_ENERGY - 1);
        energies.push_back(std::exp(log_min + fraction * (log_max - log_min)));
    }
    return energies;
}


void replace_file(const fs::path& temporary, const fs::path& destination) {
    std::error_code error;
    if (fs::exists(destination)) {
        fs::remove(destination, error);
        if (error) {
            throw std::runtime_error(
                "Could not remove old file '" + destination.string() +
                "': " + error.message()
            );
        }
    }

    fs::rename(temporary, destination, error);
    if (error) {
        throw std::runtime_error(
            "Could not rename '" + temporary.string() + "' to '" +
            destination.string() + "': " + error.message()
        );
    }
}


G4VMultipleScattering* find_proton_msc(G4ParticleDefinition* proton) {
    auto* process_manager = proton->GetProcessManager();
    if (process_manager == nullptr) {
        throw std::runtime_error("The Geant4 proton has no process manager.");
    }

    auto* processes = process_manager->GetProcessList();
    for (std::size_t index = 0; index < processes->size(); ++index) {
        auto* process = (*processes)[index];
        if (process != nullptr && process->GetProcessName() == "msc") {
            auto* msc = dynamic_cast<G4VMultipleScattering*>(process);
            if (msc == nullptr) {
                throw std::runtime_error(
                    "The proton process named 'msc' is not a "
                    "G4VMultipleScattering instance."
                );
            }
            if (!process_manager->GetProcessActivation(process)) {
                throw std::runtime_error("The proton msc process is inactive.");
            }
            return msc;
        }
    }
    throw std::runtime_error("Could not find the active proton msc process.");
}


void verify_coulomb_scat_state(
    G4ParticleDefinition* proton,
    bool expected_active
) {
    auto* process_manager = proton->GetProcessManager();
    auto* processes = process_manager->GetProcessList();
    bool found = false;

    for (std::size_t index = 0; index < processes->size(); ++index) {
        auto* process = (*processes)[index];
        if (process != nullptr && process->GetProcessName() == "CoulombScat") {
            found = true;
            if (
                process_manager->GetProcessActivation(process) !=
                expected_active
            ) {
                throw std::runtime_error(
                    "The proton CoulombScat activation state is not the "
                    "expected one."
                );
            }
        }
    }

    if (!found) {
        throw std::runtime_error("Could not find the proton CoulombScat process.");
    }
}


const G4MaterialCutsCouple* find_material_couple(const G4Material* material) {
    const auto* table = G4ProductionCutsTable::GetProductionCutsTable();
    for (std::size_t index = 0; index < table->GetTableSize(); ++index) {
        const auto* couple = table->GetMaterialCutsCouple(static_cast<G4int>(index));
        if (couple->GetMaterial() == material) {
            return couple;
        }
    }
    throw std::runtime_error(
        "Could not find the target material-cuts couple for '" +
        material->GetName() + "'."
    );
}


void verify_topas_production_cuts(const G4MaterialCutsCouple* couple) {
    const auto* cuts = couple->GetProductionCuts();
    for (const G4int index : {
        idxG4GammaCut,
        idxG4ElectronCut,
        idxG4PositronCut,
    }) {
        const G4double value = cuts->GetProductionCut(index);
        if (std::abs(value - TOPAS_PRODUCTION_CUT_M * m) > 1.0e-12 * m) {
            throw std::runtime_error(
                "The material-cuts couple does not contain the expected "
                "100 m TOPAS production cuts."
            );
        }
    }
}


void write_metadata(
    const fs::path& path,
    const std::string& material,
    const std::string& model_name,
    double polar_angle_limit_rad,
    double factor_for_angle_limit
) {
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream out(temporary);
    if (!out) {
        throw std::runtime_error("Could not write metadata: " + temporary.string());
    }

    out << "{\n"
        << "  \"geant4_version\": \"" << G4Version << "\",\n"
        << "  \"particle\": \"proton\",\n"
        << "  \"material\": \"" << material << "\",\n"
        << "  \"physics_list\": \"g4em-standard_opt4\",\n"
        << "  \"msc_process\": \"msc\",\n"
        << "  \"msc_model_class\": \"G4WentzelVIModel\",\n"
        << "  \"msc_model_name\": \"" << model_name << "\",\n"
        << "  \"wentzel_combined_mode\": true,\n"
        << "  \"msc_polar_angle_limit_rad\": "
        << std::setprecision(17) << polar_angle_limit_rad << ",\n"
        << "  \"factor_for_angle_limit\": "
        << factor_for_angle_limit << ",\n"
        << "  \"coulomb_scat_active_during_initialization\": true,\n"
        << "  \"coulomb_scat_active_during_extraction\": false,\n"
        << "  \"loss_fluctuations\": false,\n"
        << "  \"mu_had_lateral_displacement\": false,\n"
        << "  \"gamma_electron_positron_production_cut_m\": "
        << TOPAS_PRODUCTION_CUT_M << ",\n"
        << "  \"energy_min_mev\": " << ENERGY_MIN_MEV << ",\n"
        << "  \"energy_max_mev\": " << ENERGY_MAX_MEV << ",\n"
        << "  \"number_of_samples\": " << N_ENERGY << ",\n"
        << "  \"columns\": {\n"
        << "    \"energy_mev\": \"MeV\",\n"
        << "    \"first_transport_coefficient_per_m\": \"m^-1\",\n"
        << "    \"first_transport_mean_free_path_m\": \"m\"\n"
        << "  },\n"
        << "  \"geant4_api\": "
        << "\"G4VMscModel::GetTransportMeanFreePath on the active proton "
        << "G4WentzelVIModel\",\n"
        << "  \"interpretation\": "
        << "\"Reciprocal of G4VMscModel::GetTransportMeanFreePath for the "
        << "initialized Opt4 G4WentzelVIModel transport table.\"\n"
        << "}\n";

    out.close();
    replace_file(temporary, path);
}

}  // namespace


int main(int argc, char** argv) {
    std::string material_name = "<unknown>";

    try {
        const auto material_definition =
            parse_material_definition(
                argc,
                argv
            );

        material_name = material_definition.name;

        const std::string file_name =
            safe_filename(material_name);

        const fs::path output_dir =
            fs::current_path() /
            "data" /
            "wentzel_transport";

        fs::create_directories(output_dir);

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
            throw std::runtime_error("Could not obtain the Geant4 proton definition.");
        }
        if (material == nullptr) {
            throw std::runtime_error(
                "Could not obtain Geant4 material '" + material_name + "'."
            );
        }

        auto* msc = find_proton_msc(proton);
        auto* msc_model = msc->GetModelByIndex(0);
        auto* wentzel_model = dynamic_cast<G4WentzelVIModel*>(msc_model);
        if (wentzel_model == nullptr) {
            const std::string actual =
                msc_model == nullptr ? "<null>" : msc_model->GetName();
            throw std::runtime_error(
                "The active proton msc model is not G4WentzelVIModel; got '" +
                actual + "'."
            );
        }
        const std::string wentzel_model_name = wentzel_model->GetName();
        auto* em_parameters = G4EmParameters::Instance();
        if (
            em_parameters->LossFluctuation() ||
            em_parameters->MuHadLateralDisplacement()
        ) {
            throw std::runtime_error(
                "The requested TOPAS EM parameter state was not preserved "
                "through Geant4 initialization."
            );
        }
        verify_coulomb_scat_state(proton, true);

        // Match TOPAS's sequence-stage command. This removes the separate hard
        // scattering process; it does not rebuild or replace the initialized
        // combined WentzelVI MSC transport table extracted below.
        G4ProcessTable::GetProcessTable()->SetProcessActivation(
            "CoulombScat",
            "proton",
            false
        );
        verify_coulomb_scat_state(proton, false);

        G4EmCalculator calculator;
        calculator.SetVerbose(0);

        // Independent source-level check of the public calculator route:
        // G4VEmModel::Value returns E^2 times the model's directly evaluated
        // macroscopic transport cross section. It must agree with the
        // initialized interpolation table used by GetTransportMeanFreePath.
        const auto* material_couple = find_material_couple(material);
        verify_topas_production_cuts(material_couple);
        wentzel_model->SetCurrentCouple(material_couple);
        const G4double check_energy = 50.0 * MeV;
        const G4double direct_sigma_internal =
            wentzel_model->Value(material_couple, proton, check_energy) /
            (check_energy * check_energy);
        const G4double table_sigma_internal =
            calculator.GetCrossSectionPerVolume(
                check_energy,
                proton,
                "msc",
                material
            );
        const G4double check_scale = std::max(
            std::abs(direct_sigma_internal),
            std::abs(table_sigma_internal)
        );
        const G4double check_relative_error =
            check_scale > 0.0
                ? std::abs(direct_sigma_internal - table_sigma_internal) /
                    check_scale
                : DBL_MAX;
        if (
            check_scale <= 0.0 ||
            check_relative_error > 1.0e-6
        ) {
            std::ostringstream message;
            message
                << std::setprecision(17)
                << "The direct WentzelVI transport coefficient does not "
                << "agree with its initialized transport table at 50 MeV: "
                << "direct=" << direct_sigma_internal / (1.0 / m)
                << " 1/m, table=" << table_sigma_internal / (1.0 / m)
                << " 1/m, relative error=" << check_relative_error << '.';
            throw std::runtime_error(message.str());
        }
        std::cout
            << std::setprecision(12)
            << "  50 MeV direct model sigma = "
            << direct_sigma_internal / (1.0 / m) << " 1/m\n"
            << "  50 MeV table sigma        = "
            << table_sigma_internal / (1.0 / m) << " 1/m\n"
            << "  relative table error      = "
            << check_relative_error << '\n';

        const fs::path csv_path = output_dir / (file_name + ".csv");
        const fs::path temporary_csv = csv_path.string() + ".tmp";
        std::ofstream out(temporary_csv);
        if (!out) {
            throw std::runtime_error(
                "Could not write transport table: " + temporary_csv.string()
            );
        }

        out << "energy_mev,first_transport_coefficient_per_m,first_transport_mean_free_path_m\n";
        out << std::setprecision(17);

        const auto energies_mev = make_energy_grid_mev();
        for (const double energy_mev : energies_mev) {
            wentzel_model->SetCurrentCouple(material_couple);
            const G4double lambda_internal =
                wentzel_model->GetTransportMeanFreePath(
                    proton,
                    energy_mev * MeV
                );
            const double lambda_m = lambda_internal / m;
            const double sigma_per_m = 1.0 / lambda_m;

            if (
                !std::isfinite(sigma_per_m) || sigma_per_m <= 0.0 ||
                !std::isfinite(lambda_m) || lambda_m <= 0.0
            ) {
                std::ostringstream message;
                message << "Invalid WentzelVI transport value at E="
                        << energy_mev << " MeV: sigma=" << sigma_per_m
                        << " 1/m, lambda=" << lambda_m << " m.";
                throw std::runtime_error(message.str());
            }

            out << energy_mev << ',' << sigma_per_m << ',' << lambda_m << '\n';
        }

        out.close();
        replace_file(temporary_csv, csv_path);

        const fs::path metadata_path =
            output_dir / (file_name + ".metadata.json");
        write_metadata(
            metadata_path,
            material_name,
            wentzel_model_name,
            wentzel_model->PolarAngleLimit() / rad,
            em_parameters->FactorForAngleLimit()
        );

        run_manager->RunTermination();
        delete run_manager;

        std::cout
            << "Extracted WentzelVI first transport coefficient for "
            << material_name << '\n'
            << "  Geant4       = " << G4Version << '\n'
            << "  proton msc   = active\n"
            << "  model        = " << wentzel_model_name << '\n'
            << "  CoulombScat  = inactive during extraction\n"
            << "  table        = " << csv_path.string() << '\n'
            << "  metadata     = " << metadata_path.string() << '\n';
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "WentzelVI transport extraction failed for '"
            << material_name << "': " << error.what() << '\n';
        return 1;
    }
}
