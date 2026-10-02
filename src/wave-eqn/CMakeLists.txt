set(SOURCES_WAVE_EQN
    main.cpp
)

set(HEADERS_WAVE_EQN
    int/Integrator.hpp
    int/EulerIntegrator.hpp
)


set_source_files_properties(${SOURCES_WAVE_EQN} PROPERTIES LANGUAGE CUDA)

add_executable(wave-eqn ${SOURCES_WAVE_EQN} ${HEADERS_WAVE_EQN})

target_include_directories(wave-eqn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(wave-eqn PRIVATE amrex)