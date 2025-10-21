<p align="center">
    <img 
        src="https://github.com/moltools/versalign/blob/main/logo.png" 
        height="150"
    />
</p>

<p align="center">
    <a href="https://github.com/moltools/versalign/actions/workflows/tests.yml">
      <img alt="testing & quality" src="https://github.com/moltools/versalign/actions/workflows/tests.yml/badge.svg" /></a>
    <a href="https://pypi.org/project/versalign">
      <img alt="PyPI" src="https://img.shields.io/pypi/v/versalign" /></a>
    <a href="https://pypi.org/project/versalign">
      <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/versalign" /></a>
    <a href="https://github.com/moltools/versalign/blob/main/LICENSE">
      <img alt="PyPI - License" src="https://img.shields.io/pypi/l/versalign" /></a>
     <!-- <a href="https://doi.org/10.5281/zenodo.11216453">
      <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.11216453.svg" alt="DOI" /></a> -->
</p>

Versalign is a naive alignment tool for lists of arbitrary objects. Versalign is able to perform pairwise sequence alignments and star-based multiple sequence alignments, based on custom scoring functions. Versalign is primarily designed to align short-ish sequences.

Versalign is a Python library and has no command line interface.

Pairwise alignments, which is the core of this library, is built around Biopython's `PairwiseAligner` class.

## Installation

The most recent code and data can be installed directly from GitHub with:

```shell
pip install git+https://github.com/moltools/versalign.git
```

The latest stable release can be installed from PyPI with:

```shell
pip install versalign
```

Versalign has been developed for Linux and MacOS.

## Getting started

See the [examples](examples/) folder for some basic usage examples.
