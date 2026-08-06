set(SOURCES_BOXER_TEST 
    main.cpp
    AMRContainer.cpp
)
set_source_files_properties(${SOURCES_BOXER_TEST} PROPERTIES LANGUAGE CUDA)

add_executable(boxer-test ${SOURCES_BOXER_TEST} ${HEADERS_BOXERT_TEST})

# set_target_properties(boxer-test PROPERTIES
#     CUDA_STANDARD 17
#     CUDA_STANDARD_REQUIRED ON
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_RESOLVE_DEVICE_SYMBOLS ON
# )

target_link_libraries(boxer-test PRIVATE amrex Boxer::boxer)