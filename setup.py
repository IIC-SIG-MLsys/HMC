import os
import re
import shutil
from setuptools import setup

def find_and_copy_so_file(build_dir, target_dir, target_name):
    pattern = re.compile(r'^hmc\..*\.so$')

    if not os.path.isdir(build_dir):
        raise FileNotFoundError(
            f"Build directory '{build_dir}' does not exist. Run CMake build first."
        )

    candidates = [name for name in os.listdir(build_dir) if pattern.match(name)]
    if not candidates:
        raise FileNotFoundError(
            f"No pybind output like 'hmc.*.so' found under '{build_dir}'."
        )

    source_path = os.path.join(build_dir, sorted(candidates)[0])
    target_path = os.path.join(target_dir, target_name)
    shutil.copy2(source_path, target_path)
    print(f"Copied and renamed {source_path} to {target_path}")

build_directory = 'build'  # where the .so file is located
pkg_directory = 'hmc'  # target directory for the .so file
new_filename = 'hmc.so'

if not os.path.exists(pkg_directory):
    os.makedirs(pkg_directory)
    print(f"Created directory: {pkg_directory}")
find_and_copy_so_file(build_directory, pkg_directory, new_filename)

setup(
    name='hmc',
    version='0.0.1',
    author='JaceLau',
    author_email='jacalau@outlook.com',
    description='HMC python binding',
    packages=['hmc'],
    package_data={
        'hmc': ['hmc.so'],
    },
    include_package_data=True,
    zip_safe=False,
)
