import os
import sys
import setuptools

with open("./README.md", "r") as fh:
    long_description = fh.read()

req_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')

install_requires = []
if os.path.isfile(req_path):
    with open(req_path) as f:
        install_requires = f.read().splitlines()
else:
    print("requirements file not found. Exit...")
    sys.exit()

setuptools.setup(
    name="dfagent",
    version="0.1",
    author="Luigi Di Sotto, Diego Giorgini",
    author_email="luigi.disotto@aitechnologies.it, diego.giorgini@aitechnologies.it",
    description="Dialogflow agent is a library for online or offline handling of Dialogflow agents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aitechnologies-it/dialogflow-agent",
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=install_requires
)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()