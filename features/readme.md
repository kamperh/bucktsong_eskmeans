Feature Extraction
==================

Overview
--------
To extract MFCC and filterbank features, run:

    ./extract_features_buckeye.py
    ./extract_features_xitsonga.py

The rest of this document describes some of the feature sets and file formats.


Buckeye sets
------------
Buckeye is divided into a number of sets based on the speakers:

- sample: s2801a, s2801b, s2802a, s2802b, s2803a, s3701a, s3701b, s3702a,
  s3702b, s3703a, s3703b
- devpart1 (used for training): s02, s04, s05, s08, s12, s16, s03, s06, s10,
  s11, s13, s38
- devpart2 (used for validation): s18, s17, s37, s39, s19, s22, s40, s34
- ZS (used for testing): s20, s25, s27, s01, s26, s31, s29, s23, s24, s32, s33,
  s30


NumPy archive key format
------------------------
For the unsegmented feature NumPy archives (e.g. `mfcc/buckeye/zs.dd.npz`),
dictionary keys look as follows:

    s01_01a_003222-003255

For speaker `s01`, this is utterance `01a` from frames `3222` to `3255`.
