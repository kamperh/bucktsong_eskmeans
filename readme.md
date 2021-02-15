Embedded Segmental K-Means Applied to Buckeye English and NCHLT Xitsonga
========================================================================

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/kamperh/eskmeans/blob/master/license.md)


Overview
--------
Unsupervised acoustic word segmentation and clustering of Buckeye English and
NCHLT Xitsonga data using the embedded segmental K-means (ES-KMeans) algorithm.
The experiments are described in:

> H. Kamper, K. Livescu, and S. J. Goldwater, "An embedded segmental K-means
> model for unsupervised segmentation and clustering of speech," in *Proc.
> ASRU*, 2017. [[arXiv](https://arxiv.org/abs/1703.08135)]

Please cite this paper if you use the code.

This recipe relies on the separate
[ES-KMeans](https://github.com/kamperh/eskmeans/) package, which performs the
actual unsupervised segmentation and clustering.


Download datasets
-----------------
The Buckeye English and portions of the NCHLT Xitsonga corpora are used:

- Buckeye corpus: http://buckeyecorpus.osu.edu/
- NCHLT Xitsonga portion: http://zerospeech.com/2015/index.html

From the complete Buckeye corpus we split off several subsets. The most
important are the sets labelled as `devpart1` and `zs`. These sets respectively
correspond to `English1` and `English2` in [(Kamper et al.,
2016)](http://arxiv.org/abs/1606.06950).


Install dependencies
--------------------
Dependencies can be installed in a conda environment:

    conda env create -f environment.yml
    conda activate eskmeans

Install the [ES-KMeans](https://github.com/kamperh/eskmeans/) package:

    mkdir ../src/
    git clone https://github.com/kamperh/eskmeans.git ../src/eskmeans/


Extract speech features
-----------------------
Extract MFCCs in `features/` as follows:

    cd features/
    ./extract_features_buckeye.py
    ./extract_features_xitsonga.py

More details on the feature file formats are given in
[features/readme.md](features/readme.md).


Unsupervised syllable boundary detection
----------------------------------------
As a preprocessing step, we constrain the allowed word boundary positions to
boundaries detected by an unsupervised syllable boundary detection algorithm.
We specifically use the algorithm described in:

> O. J. Räsänen, G. Doyle, and M. C. Frank, "Pre-linguistic segmentation of
> speech into syllable-like units," *Cognition*, 2018.

Extract the syllable boundaries in `syllables/` as follows:

    cd syllables/
    ./get_syl_landmarks.py buckeye
    ./get_syl_landmarks.py xitsonga


Downsampled acoustic word embeddings
------------------------------------
Extract and evaluate downsampled acoustic word embeddings by running the steps
in [downsample/readme.md](downsample/readme.md).


ES-KMeans: Segmentation and clustering
--------------------------------------
Segmentation and clustering is performed using the
[ES-KMeans](https://github.com/kamperh/eskmeans/) package. Run the steps in
[segmentation/readme.md](segmentation/readme.md).


Contributors
------------
- [Herman Kamper](http://www.kamperh.com/)
- [Karen Livescu](http://ttic.uchicago.edu/~klivescu/)
- [Sharon Goldwater](http://homepages.inf.ed.ac.uk/sgwater/)
