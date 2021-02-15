Segmentation and Clustering
===========================

Data pre-processing
-------------------
Format the Buckeye English and NCHLT Xitsonga data into the input format used
by ES-KMeans:

    # Downsampled MFCCs
    ./get_data_downsample.py devpart1 unsup_syl mfcc 10
    ./get_data_downsample.py zs unsup_syl mfcc 10
    ./get_data_downsample.py xitsonga unsup_syl mfcc 10

Get a subset of the data for a particular speaker:

    ./get_data_speaker.py data/devpart1/mfcc.n_10.unsup_syl/ s38

Get data for all the individual speakers:

    ./get_data_sd.py devpart1 unsup_syl mfcc 10
    ./get_data_sd.py zs unsup_syl mfcc 10
    ./get_data_sd.py xitsonga unsup_syl mfcc 10


Single-speaker segmentation and evaluation
------------------------------------------
Perform unsupervised acoustic segmentation for a specific speaker using MFCC
embeddings and evaluate:

    # Downsampled Devpart1 s38 MFCCs
    ./ksegment.py data/devpart1/mfcc.n_10.unsup_syl/s38 --min_duration 20
    ./segment_eval.py kmodels/devpart1/mfcc.n_10.unsup_syl/s38/085e46cd21/ksegment.pkl

A selection of evaluation output:

    NED: 0.8629902880246759 (23991 pairs)
    uWER: 0.8539341138034304
    uWER_many: 0.7270623468554315
    Word boundaries:
    tolerance = one phone: P = 0.7688888888888888, R = 0.4974838245866283, F = 0.6041030117852466
    Token scores:
    tolerance = one phone: P = 0.4336677814938685, R = 0.31772393139123334, F = 0.36675047140163425


Speaker-dependent segmentation and evaluation
---------------------------------------------
Perform unsupervised acoustic segmentation for all speakers by spawning
parallel jobs:

    # Devpart1
    stdbuf -oL ./spawn_ksegment_sd.py data/devpart1/mfcc.n_10.unsup_syl --min_duration 20 --segment_n_iter 10
    ./spawn_segment_sd_eval.py kmodels/devpart1/mfcc.n_10.unsup_syl/sd_e333d139e0/models.txt

    # ZS (test)
    stdbuf -oL ./spawn_ksegment_sd.py data/zs/mfcc.n_10.unsup_syl --min_duration 20 --segment_n_iter 10
    ./spawn_segment_sd_eval.py kmodels/zs/mfcc.n_10.unsup_syl/sd_8acab507e2/models.txt

    # Xitsonga
    stdbuf -oL ./spawn_ksegment_sd.py data/xitsonga/mfcc.n_10.unsup_syl --min_duration 25 --segment_n_iter 10
    ./spawn_segment_sd_eval.py kmodels/xitsonga/mfcc.n_10.unsup_syl/sd_66e4b282b2/models.txt

A selection of evaluation output for devpart1:

    Avg. clustering average purity: 0.42922186506499443
    NED: 0.8272808743247441
    uWER: 0.8596692721981245
    uWER_many: 0.7227952409094457
    Word boundaries:
    tolerance = one phone: P = 0.7515040743279263, R = 0.45349264705882353, F = 0.5656473015964002
    Word token scores:
    tolerance = one phone: P = 0.4196632912186099, R = 0.29852477113323894, F = 0.3488776673725362

