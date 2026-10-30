cmake_minimum_required(VERSION 3.16)

project(geant4_material_extractors LANGUAGES CXX)

find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

add_executable(
    geant4_material_export
    geant4_material_export.cc
)

add_executable(
    geant4_wentzel_transport_export
    geant4_wentzel_transport_export.cc
)

target_compile_features(
    geant4_material_export
    PRIVATE cxx_std_17
)

target_compile_features(
    geant4_wentzel_transport_export
    PRIVATE cxx_std_17
)

target_link_libraries(
    geant4_material_export
    PRIVATE ${Geant4_LIBRARIES}
)

target_link_libraries(
    geant4_wentzel_transport_export
    PRIVATE ${Geant4_LIBRARIES}
)