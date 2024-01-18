# Outlier Detection in Heterogeneous Datasets using Automatic Tuple Expansion

### Clément Pit-Claudel, Zelda Mariet, Rachael Harding, Samuel Madden

## Abstract

Rapidly developing areas of information technology are generating massive amounts of data. Human errors, sensor failures, and other unforeseen circumstances unfortunately tend to undermine the quality and consistency of these datasets by introducing outliers -- data points that exhibit surprising behavior when compared to the rest of the data. Characterizing, locating, and in some cases eliminating these outliers offers interesting insight about the data under scrutiny and reinforces the confidence that one may have in conclusions drawn from otherwise noisy datasets.

In this paper, we describe a tuple expansion procedure which reconstructs rich information from semantically poor SQL data types such as strings, integers, and floating point numbers.
We then use this procedure as the foundation of a new user-guided outlier detection framework, dBoost, which relies on inference and statistical modeling of heterogeneous data to flag suspicious fields in database tuples. This repository contains our implementation, publicly available under version 3 of the GNU General Public License.

## Getting started

After cloning the repository, run `./dboost-stdin.py -h` in the `dboost` directory to show the documentation.
