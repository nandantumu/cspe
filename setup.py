from setuptools import find_packages, setup

package_name = "cspe"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "clip @ git+https://github.com/openai/CLIP.git",
        "PIT @ git+https://github.com/nandantumu/PIT.git",
        "ftfy",
        "regex",
        "tqdm",
    ],
    zip_safe=True,
    maintainer="Nandan Tumu",
    maintainer_email="nandant@nandantumu.com",
    description="This package generates Context Specific Parameter Estimates",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "context_identification=cspe.scripts.context_identifier:main"
        ],
    },
)
