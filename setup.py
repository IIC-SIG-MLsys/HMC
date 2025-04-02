from setuptools import setup
import os, re, shutil

def find_and_copy_so_file(build_dir, target_dir, target_name):
    pattern = re.compile(r'^hmc\..*\.so$')
    
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
        print(f"Created directory: {build_dir}")

    for filename in os.listdir(build_dir):
        if pattern.match(filename):
            source_path = os.path.join(build_dir, filename)
            target_path = os.path.join(target_dir, target_name)

            shutil.copy(source_path, target_path)
            print(f"Copied and renamed {source_path} to {target_path}")
            return

    print("No matching .so file found in the build directory.")

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
        'hmc': ['hmc/hmc.so'],
    },
    include_package_data=True,
    zip_safe=False,
)