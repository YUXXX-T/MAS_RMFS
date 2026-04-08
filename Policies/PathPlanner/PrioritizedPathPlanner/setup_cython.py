"""
构建 Cython A* 扩展模块。

用法:
    cd f:\\MAS_RMFS\\Policies\\PathPlanner\\PrioritizedPathPlanner
    python setup_cython.py build_ext --inplace

编译成功后会生成 _astar_core.cp311-win_amd64.pyd (Windows)
或 _astar_core.cpython-311-x86_64-linux-gnu.so (Linux)。
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "_astar_core",
        sources=["_astar_core.pyx"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=["/O2"] if __import__("sys").platform == "win32" else ["-O3"],
    )
]

setup(
    name="astar_core",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    ),
)
