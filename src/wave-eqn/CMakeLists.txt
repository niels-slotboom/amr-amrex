set(SOURCES_WAVE_EQN
    main.cpp
)

set(HEADERS_WAVE_EQN
    Integrator.hpp
)


set_source_files_properties(${SOURCES_WAVE_EQN} PROPERTIES LANGUAGE CUDA)

add_executable(wave-eqn ${SOURCES_WAVE_EQN} ${HEADERS_WAVE_EQN})

target_link_libraries(wave-eqn PRIVATE amrex)