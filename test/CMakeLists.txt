add_executable(symla_smoke_test unit/smoke_test.cpp)
target_link_libraries(symla_smoke_test PRIVATE symla::symla)
add_test(NAME smoke_test COMMAND symla_smoke_test)

add_executable(symla_ordering_test unit/ordering_test.cpp)
target_link_libraries(symla_ordering_test PRIVATE symla::symla)
target_include_directories(symla_ordering_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME ordering_test COMMAND symla_ordering_test)

add_executable(symla_elimination_tree_test unit/elimination_tree_test.cpp)
target_link_libraries(symla_elimination_tree_test PRIVATE symla::symla)
target_include_directories(symla_elimination_tree_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME elimination_tree_test COMMAND symla_elimination_tree_test)

add_executable(symla_symbolic_test unit/symbolic_test.cpp)
target_link_libraries(symla_symbolic_test PRIVATE symla::symla)
target_include_directories(symla_symbolic_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
add_test(NAME symbolic_test COMMAND symla_symbolic_test)
