from setuptools import setup, find_packages
from os import path
import versioneer

cur_dir = path.abspath(path.dirname(__file__))

setup(
    name="aiera-assistant",
    description="Aiera Chat Engine",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(exclude=["bin", "sandbox"]),
    include_package_data=True,
    python_requires=">=3.10",
)
