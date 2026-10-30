#pragma once

#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


struct MaterialDefinition {
    std::string name;
    bool custom = false;
    double density_g_cm3 = 0.0;
    std::vector<std::pair<std::string, double>> mass_fractions;
};


inline MaterialDefinition parse_material_definition(
    int argc,
    char** argv
) {
    // Existing NIST-material mode:
    //   exporter G4_WATER
    if (argc == 2) {
        return MaterialDefinition{
            argv[1],
            false,
            0.0,
            {},
        };
    }

    // Custom-material mode:
    //   exporter --custom NAME DENSITY_G_CM3 \
    //       Hydrogen 0.1 Oxygen 0.9
    if (
        argc < 6 ||
        std::string(argv[1]) != "--custom" ||
        (argc - 4) % 2 != 0
    ) {
        throw std::runtime_error(
            "Expected either:\n"
            "  G4_MATERIAL_NAME\n"
            "or:\n"
            "  --custom NAME DENSITY_G_CM3 "
            "ELEMENT FRACTION [ELEMENT FRACTION ...]"
        );
    }

    MaterialDefinition definition;
    definition.name = argv[2];
    definition.custom = true;
    definition.density_g_cm3 = std::stod(argv[3]);

    if (
        !std::isfinite(definition.density_g_cm3) ||
        definition.density_g_cm3 <= 0.0
    ) {
        throw std::runtime_error(
            "Custom material density must be positive and finite."
        );
    }

    double fraction_sum = 0.0;

    for (int i = 4; i < argc; i += 2) {
        const std::string element_name = argv[i];
        const double fraction = std::stod(argv[i + 1]);

        if (!std::isfinite(fraction) || fraction < 0.0) {
            throw std::runtime_error(
                "Custom material mass fractions must be finite and non-negative."
            );
        }

        if (fraction == 0.0) {
            continue;
        }

        definition.mass_fractions.emplace_back(
            element_name,
            fraction
        );

        fraction_sum += fraction;
    }

    if (definition.mass_fractions.empty()) {
        throw std::runtime_error(
            "Custom material contains no non-zero elemental fractions."
        );
    }

    if (std::abs(fraction_sum - 1.0) > 1.0e-8) {
        throw std::runtime_error(
            "Custom material mass fractions must sum to 1."
        );
    }

    return definition;
}


// TOPAS Schneider parameter files use full element names, while
// G4NistManager::FindOrBuildElement expects the chemical symbol.
// Accept both so the exporter can be fed directly from TOPAS data.
inline std::string nist_element_symbol(
    const std::string& element_name
) {
    if (element_name == "Hydrogen")   return "H";
    if (element_name == "Carbon")     return "C";
    if (element_name == "Nitrogen")   return "N";
    if (element_name == "Oxygen")     return "O";
    if (element_name == "Magnesium")  return "Mg";
    if (element_name == "Phosphorus") return "P";
    if (element_name == "Sulfur")     return "S";
    if (element_name == "Chlorine")   return "Cl";
    if (element_name == "Argon")      return "Ar";
    if (element_name == "Calcium")    return "Ca";
    if (element_name == "Sodium")     return "Na";
    if (element_name == "Potassium")  return "K";
    if (element_name == "Titanium")   return "Ti";

    // If the caller already supplied a symbol, or a future element is
    // introduced, let Geant4 try the supplied value directly.
    return element_name;
}


inline G4Material* build_material(
    const MaterialDefinition& definition
) {
    auto* nist = G4NistManager::Instance();

    if (!definition.custom) {
        return nist->FindOrBuildMaterial(
            definition.name
        );
    }

    if (
        auto* existing = G4Material::GetMaterial(
            definition.name,
            false
        )
    ) {
        return existing;
    }

    auto* material = new G4Material(
        definition.name,
        definition.density_g_cm3 * g / cm3,
        static_cast<G4int>(
            definition.mass_fractions.size()
        )
    );

    for (
        const auto& [element_name, fraction] :
            definition.mass_fractions
    ) {
        const std::string symbol =
            nist_element_symbol(
                element_name
            );

        auto* element =
            nist->FindOrBuildElement(
                symbol
            );

        if (element == nullptr) {
            throw std::runtime_error(
                "Could not construct Geant4 element '" +
                element_name +
                "' (symbol '" +
                symbol +
                "')."
            );
        }

        material->AddElement(
            element,
            fraction
        );
    }

    return material;
}
