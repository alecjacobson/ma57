add_executable(symla_smoke_test unit/smoke_test.cpp)
target_link_libraries(symla_smoke_test PRIVATE symla::symla)
add_test(NAME smoke_test COMMAND symla_smoke_test)
