Downsampled Acoustic Word Embeddings
====================================

Dense embedding extraction on unsupervised syllable landmarks
-------------------------------------------------------------
First perform the unsupervised syllable segmentation in `../syllables/`. Then
link the landmarks from the unsupervised syllable segmentation:

    mkdir -p output/devpart1/
    ln -s ~/endgame/projects/stellenbosch/bucktsong_eskmeans/syllables/landmarks/devpart1/landmarks.unsup_syl.pkl output/devpart1/

    mkdir -p output/zs/
    ln -s ~/endgame/projects/stellenbosch/bucktsong_eskmeans/syllables/landmarks/zs/landmarks.unsup_syl.pkl output/zs/

    mkdir -p output/xitsonga/
    ln -s ~/endgame/projects/stellenbosch/bucktsong_eskmeans/syllables/landmarks/xitsonga/landmarks.unsup_syl.pkl output/xitsonga/

Get the segmentation intervals over the landmarks:

    ./get_seglist.py devpart1 unsup_syl
    ./get_seglist.py zs unsup_syl
    ./get_seglist.py xitsonga unsup_syl

Get the dense embeddings over the segmentation intervals:

    ./downsample_dense.py --frame_dims 13 devpart1 unsup_syl mfcc
    ./downsample_dense.py --frame_dims 13 zs unsup_syl mfcc
    ./downsample_dense.py --frame_dims 13 xitsonga unsup_syl mfcc

The remainder of this readme describes intermediate evaluation steps and gives
more details on the file formats. These steps do not have to be completed to
run the rest of the recipe in `../readme.md`.


Isolated words and intermediate same-different evaluation
---------------------------------------------------------
Perform downsampling on MFCCs without deltas:

    # Devpart1 downsampling
    mkdir -p output/devpart1/
    n_samples=10
    ./downsample.py --technique resample --frame_dims 13 ../features/mfcc/buckeye/devpart1.samediff.dd.npz   output/devpart1/samediff.mfcc.downsample_${n_samples}.npz ${n_samples}

    # Devpart1 same-different evaluation
    ./eval_samediff.py --mvn output/devpart1/samediff.mfcc.downsample_${n_samples}.npz

    # Xitsonga downsampling
    mkdir -p output/xitsonga/
    n_samples=10
    ./downsample.py --technique resample --frame_dims 13 ../features/mfcc/xitsonga/xitsonga.samediff.dd.npz output/xitsonga/samediff.mfcc.downsample_${n_samples}.npz ${n_samples}

    # Xitsonga same-different evaluation
    ./eval_samediff.py --mvn output/xitsonga/samediff.mfcc.downsample_${n_samples}.npz

The average precision should be as follows:

    Devpart1, downsampled MFCCs: 0.20536592253939154
    Xitsonga, downsampled MFCCs: 0.11326389791908284


Description of the segmentation interval dictionary
---------------------------------------------------
In each pickled file (e.g. `devpart1.seglist.unsup_syl.pkl`) is a single
dictionary containing a segmentation list: all the intervals for which
embeddings are required. You can load this dictionary as:

    import pickle
    with open("devpart1.seglist.unsup_syl.pkl", "rb") as f:
        seglist = pickle.load(f)

Each of the keys in the `seglist` dictionary looks like "s0403b_031097-031250".
This key corresponds to a utterance in the corpus. This utterance is from the
Buckeye audio file s0403b with the speech from frame 31097 to 31250
(inclusive), i.e. the speech from 310.97 to 312.50 seconds.

The entry in the `seglist` dictionary for this key will specify the portions of
speech for which I need embeddings. As an example,
`seglist["s0403b_031097-031250"]` looks as follows:

    [(0, 12),
     (0, 33),
     (0, 63),
     (0, 86),
     (0, 153),
     (12, 33),
     (12, 63),
     (12, 86),
     (12, 153),
     (33, 63),
     (33, 86),
     (33, 153),
     (63, 86),
     (63, 153),
     (86, 153)]

This means that for utterance "s0403b_031097-031250", the first embedding I
need stretches from frame 0 to frame 12 (inclusive) within this utterance.
Working this back to the original audio, this will correspond to the interval
(31097 + 0, 31097 + 12) = (31097, 31109) in the original audio file "s0403b".

If you want to use arbitrary acoustic word embeddings, you would need to
construct a dictionary `embeddings` having the same keys as `seglist` with
embeddings in the same order as above. As an example, if we have embeddings of
dimensionality 3, the dictionary `embeddings["s0403b_031097-031250"]` should be
a matrix looking something like this:

    array([[ 0.53194974,  0.77044634,  1.79343371],
           [ 0.89781123, -0.01795841, -0.09023884],
           [ 0.85071463, -0.90321776, -0.84269961],
           [ 1.13889494,  0.04895219,  0.16259716],
           [-1.1104843 , -1.08810261,  1.60176995],
           [ 0.935693  ,  1.9002663 ,  0.34357188],
           [-0.06871111, -0.09815952,  0.41243265],
           [-0.52154948, -0.79067086,  0.58134061],
           [-0.29175393, -0.28265766,  0.23767037],
           [-1.01458858,  0.84830074,  0.68321245],
           [-0.26042844,  1.07567039, -1.26573127],
           [-1.27184952, -0.53725344,  0.50796913],
           [-0.74591164, -0.30181373, -0.41273143],
           [-0.6164348 , -0.66225628, -0.02732791],
           [ 1.27053322,  0.03837404,  0.8739168 ]])

So, for "s0403b_031097-031250" the embedding for speech frames (0, 12) is
`[ 0.53194974,  0.77044634,  1.79343371]` while the embedding for frames
(86, 153) is `[ 1.27053322,  0.03837404,  0.8739168 ]`.
