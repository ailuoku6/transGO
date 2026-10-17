import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

KATAGO_RULE_SOURCES = [
    "cpp/vendor/katago/core/bsearch.cpp",
    "cpp/vendor/katago/core/global.cpp",
    "cpp/vendor/katago/core/hash.cpp",
    "cpp/vendor/katago/core/rand.cpp",
    "cpp/vendor/katago/core/rand_helpers.cpp",
    "cpp/vendor/katago/core/md5.cpp",
    "cpp/vendor/katago/core/sha2.cpp",
    "cpp/vendor/katago/core/timer.cpp",
    "cpp/vendor/katago/game/board.cpp",
    "cpp/vendor/katago/game/boardhistory.cpp",
    "cpp/vendor/katago/game/rules.cpp",
]

setup(
    ext_modules=[Pybind11Extension(
        "transformer_go._native", ["cpp/native.cpp", *KATAGO_RULE_SOURCES],
        cxx_std=17, extra_compile_args=["-O3", "-Wall", "-Wextra"] +
        (["-mmacosx-version-min=14.0"] if sys.platform == "darwin" else []),
        extra_link_args=["-mmacosx-version-min=14.0"] if sys.platform == "darwin" else [],
    )],
    cmdclass={"build_ext": build_ext},
)
