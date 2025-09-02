# pyNega2

TODO: Summary

## ToDo List

- Python scripts and modules have been updated to Python 3.X but still need proper structuring and validation.
- Setup of the project may still need minor changes like missing requirements.

## Installation

#### From ICARUS repository

To install `pyNega2` as a package in your Python (virtual) environment use `pip` to install it directly from GitLab:

```bash
pip install git+https://gitlab.upc.edu/icarus/projects/pynega2
```

#### Editable local project

If you want to install it from a local repository, do the following steps to link the package to the corresponding
folder so that any changes will reflect in the installed package:

```bash
# Download the repository
git clone git@gitlab.upc.edu:icarus/projects/pynega2.git

# Install from folder with editable flag
pip install -e ./pynega2/
```

## Development

To get the source code, clone it from the UPC GitLab's `ICARUS` repository:

```bash
# Download the repository
git clone git@gitlab.upc.edu:icarus/projects/pynega2.git

# Move into the pyBADA project folder
cd ./pynega2/
```

Configure a virtual environment in Python3.X for the project and call `pip install -r requirements.txt` to install all
necessary packages for the project to work. You can also install `pyNega2` as an
[editable package](README.md#Editable-local-project) if you want to test it in other projects.

You may need the following system packages to fully install the above requirements:

```shell
sudo apt install g++ mpi-default-dev
```

## Changelog

See [CHANGELOG](CHANGELOG.md) file.

## Authors

- Project Manager: Dr. Xavier Prats.
- Chief software architect: Dr. David de la Torre.
- Developers (in alphabetical order):
    - Cabrera, Bryan Gustavo
    - Dalmau, Ramon
    - de la Torre, David
    - Eritja, Antoni
    - Melgosa, Marc
    - Prat, Martí
    - Rad, Ioan Octavian
    - Sáez, Raúl
    - Vilardaga, Santi

## Citation

TODO

## License

All rights reserved.

See [LICENSE](LICENSE) file.