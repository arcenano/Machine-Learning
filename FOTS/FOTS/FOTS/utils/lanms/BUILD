package(default_visibility = ["//visibility:public"])

#load("//bazel:build_defs.bzl", "pybind_extension", "pybind_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "adaptor",  # This name is not actually created!
    srcs = [ "lanms.h", "adaptor.cpp", "clipper.hpp", "clipper.cpp" ],
)

py_library(
    name = "adaptor",
    data = [":adaptor.so"],
)

py_test(
    name = "example_test",
    srcs = ["example_test.py"],
    deps = [
        ":adaptor"
    ],
)

# cc_library(
#         name = "adaptor",
#         srcs = ["adaptor.cpp"],
#         copts = [
#             "-fexceptions",
#         ],
#         features = [
#             "-use_header_modules",  # Required for pybind11.
#             "-parse_headers",
#         ],
#         deps = [ 
#             "@pybind11",
#             "@local_config_python//:python_headers",
#             ":clipper",
#         ]
#     )
