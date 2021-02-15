#!/usr/bin/env python

"""
ES-KMeans applied to Buckeye English and NCHLT Xitsonga.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015-2017, 2021
"""

from datetime import datetime
from os import path
import argparse
import hashlib
import numpy as np
import os
import pickle
import random
import sys

sys.path.append(os.path.abspath(path.join("..", "..", "src", "eskmeans")))

from eskmeans import eskmeans_wordseg


#-----------------------------------------------------------------------------#
#                           DEFAULT TRAINING OPTIONS                          #
#-----------------------------------------------------------------------------#

default_options_dict = {
    "data_dir": None,                 # "data/devpart1/mfcc.unsup_syl",
    "model_dir": "kmodels",
    "K_max": "0.2landmarks",          # can either be an absolute number or
                                      # "0.2landmarks", indicating a proportion
    "init_am_n_iter": 0,              # initial iterations of acoustic model
                                      # sampling; can also be "kmeans"
    "init_am_assignments": "spread",  # "rand", "spread"
    "p_boundary_init": 1.0,           # probability of initial boundaries
    "n_slices_min": 0,
    "n_slices_max": 6,
    "min_duration": 0,                # minimum number of frames in segment
    "segment_n_iter": 10,
    "n_iter_inbetween_kmeans": 0,
    "rnd_seed": 42,
    "wip": 0,
    "n_cpus": 1,
    "n_batches": 1,
    }


#-----------------------------------------------------------------------------#
#                            SEGMENTATION FUNCTIONS                           #
#-----------------------------------------------------------------------------#

def ksegment(options_dict, suppress_pickling=False):
    """
    Segment and save the results of k-segmental acoustic word segmentation.

    The `suppress_pickling` parameter is useful for initializing a model from
    previous output.
    """

    print(datetime.now())

    random.seed(options_dict["rnd_seed"])
    np.random.seed(options_dict["rnd_seed"])

    # Set output pickle filename
    hasher = hashlib.md5(repr(sorted(options_dict.items())).encode("ascii"))
    hash_str = hasher.hexdigest()[:10]
    model_dir = path.join(options_dict["model_dir"], hash_str)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    options_dict_fn = path.join(model_dir, "options_dict.pkl")
    if not suppress_pickling:
        print("Writing: " + options_dict_fn)
        with open(options_dict_fn, "wb") as f:
            pickle.dump(options_dict, f, -1)
    print("Options: " + str(options_dict))

    print("Reading from directory: " + options_dict["data_dir"])
    landmarks_dict_fn = path.join(
        options_dict["data_dir"], "landmarks.pkl"
        )
    dense_embeddings_fn = path.join(
        options_dict["data_dir"], "dense_embeddings.npz"
        )
    vec_ids_dict_fn = path.join(
        options_dict["data_dir"], "vec_ids.pkl"
        )
    durations_dict_fn = path.join(
        options_dict["data_dir"], "durations.pkl"
        )
    word_gt_dict_fn = path.join(
        options_dict["data_dir"], "word_gt.pkl"
        )
    # phone_gt_dict_fn = path.join(data_dir, "phone_gt.pkl")
    with open(landmarks_dict_fn, "rb") as f:
        landmarks_dict = pickle.load(f)
    dense_embeddings = dict(np.load(dense_embeddings_fn))
    with open(vec_ids_dict_fn, "rb") as f:
        vec_ids_dict = pickle.load(f)
    with open(durations_dict_fn, "rb") as f:
        durations_dict = pickle.load(f)
    with open(word_gt_dict_fn, "rb") as f:
        word_gt_bound_dict = pickle.load(f)
        word_gt_label_dict = pickle.load(f)
    print("No. of utterances: " + str(len(landmarks_dict)))

    n_landmarks = sum([len(i) for i in landmarks_dict.values()])
    print("No. of landmarks: " + str(n_landmarks))
    if "landmarks" in str(options_dict["K_max"]):
        # The number of components are set as a proportion of landmarks
        proportion = float(options_dict["K_max"].replace("landmarks", ""))
        K_max = int(np.floor(proportion * n_landmarks))
    else:
        K_max = int(options_dict["K_max"])
    print("K_max: " + str(K_max))

    D = dense_embeddings[list(dense_embeddings)[0]].shape[1]
    print("Embedding dimensionality: " + str(D))

    print(datetime.now())
    print("Normalizing embeddings")
    n_embeds = 0
    for utt in dense_embeddings:
        for i in range(dense_embeddings[utt].shape[0]):
            n_embeds += 1
            cur_embed = dense_embeddings[utt][i, :]
            norm = np.linalg.norm(cur_embed)
            assert norm != 0.
            dense_embeddings[utt][i, :] = cur_embed / np.linalg.norm(cur_embed)
    print("No. of embeddings: " + str(n_embeds))

    # Setup model
    print("Setting up model")
    ksegmenter = eskmeans_wordseg.ESKmeans(
        K_max=K_max,
        embedding_mats=dense_embeddings, vec_ids_dict=vec_ids_dict,
        durations_dict=durations_dict, landmarks_dict=landmarks_dict,
        p_boundary_init=options_dict["p_boundary_init"],
        n_slices_min=options_dict["n_slices_min"],
        n_slices_max=options_dict["n_slices_max"],
        min_duration=options_dict["min_duration"],
        init_assignments=options_dict["init_am_assignments"],
        wip=options_dict["wip"]
        )

    # Initialize acoustic model by training for a few iterations
    if options_dict["init_am_n_iter"] > 0:
        print("Performing initial K-means iterations")
        am_init_record = ksegmenter.acoustic_model.fit(
            options_dict["init_am_n_iter"], consider_unassigned=False
            )
        if not suppress_pickling:
            fn = path.join(model_dir, "am_init_record_dict.pkl")
            print("Writing: " + fn)
            with open(fn, "wb") as f:
                pickle.dump(am_init_record, f, -1)

    # Perform segmentation
    if options_dict["segment_n_iter"] > 0:
        if options_dict["n_cpus"] > 1:
            segmenter_record = ksegmenter.segment_parallel(
                options_dict["segment_n_iter"],
                options_dict["n_iter_inbetween_kmeans"],
                n_cpus=options_dict["n_cpus"],
                n_batches=options_dict["n_batches"]
                )
        else:
            segmenter_record = ksegmenter.segment(  
                options_dict["segment_n_iter"],
                options_dict["n_iter_inbetween_kmeans"]
                )

        if not suppress_pickling:
            fn = path.join(model_dir, "segmenter_record_dict.pkl")
            print("Writing: " + fn)
            with open(fn, "wb") as f:
                pickle.dump(segmenter_record, f, -1)

    # Obtain clusters and landmarks (frame indices)
    unsup_transcript = {}
    unsup_landmarks = {}
    unsup_landmark_indices = {}
    for i_utt in range(ksegmenter.utterances.D):
        utt = ksegmenter.ids_to_utterance_labels[i_utt]
        unsup_transcript[utt] = ksegmenter.get_unsup_transcript_i(i_utt)
        if -1 in unsup_transcript[utt]:
            logger.warning(
                "Unassigned cuts in: " + utt + " (transcript: " +
                str(unsup_transcript[utt]) + ")"
                )
        unsup_landmarks[utt] = (
            ksegmenter.utterances.get_segmented_landmarks(i_utt)
            )
        unsup_landmark_indices[utt] = (
            ksegmenter.utterances.get_segmented_landmark_indices(i_utt)
            )

    # Write output
    if not suppress_pickling:
        output_fn = path.join(model_dir, "ksegment.pkl")
        print("Writing: " + output_fn)
        with open(output_fn, "wb") as f:
            pickle.dump(unsup_landmarks, f, -1)
            pickle.dump(unsup_transcript, f, -1)
            if (options_dict["init_am_n_iter"] != "kmeans") and (
                    options_dict["init_am_n_iter"] > 0):
                pickle.dump(am_init_record, f, -1)
            else:
                pickle.dump(None, f, -1)
            if options_dict["segment_n_iter"] > 0:
                pickle.dump(segmenter_record, f, -1)
            else:
                pickle.dump(None, f, -1)
            pickle.dump(unsup_landmark_indices, f, -1)
        print(datetime.now())
        return output_fn


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "data_dir", type=str, help="data directory"
        )
    parser.add_argument(
        "--wip", type=float, help="default: %(default)s",
        default=default_options_dict["wip"]
        )
    parser.add_argument(
        "--rnd_seed", type=int, help="default: %(default)s",
        default=default_options_dict["rnd_seed"]
        )
    parser.add_argument(
        "--min_duration", type=int, help="default: %(default)s",
        default=default_options_dict["min_duration"]
        )
    parser.add_argument(
        "--init_am_n_iter", type=int, help="default: %(default)s",
        default=default_options_dict["init_am_n_iter"]
        )
    parser.add_argument(
        "--segment_n_iter", type=int, help="default: %(default)s",
        default=default_options_dict["segment_n_iter"]
        )
    parser.add_argument(
        "--eval", help="evaluate output from segmentation", action="store_true"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def eval_pickle(pickle_fn):
    import segment_eval
    print("Evaluating: " + pickle_fn)
    output_fn = path.join(path.split(pickle_fn)[0], "segment_eval.txt")
    print("Writing: " + output_fn)
    with open(output_fn, "w") as f:
        sys.stdout = f
        result_dict = segment_eval.segment_eval(
            pickle_fn, "word", suppress_plot=True
            )
    sys.stdout = sys.__stdout__
    result_dict_fn = path.join(
        path.split(pickle_fn)[0], "segment_eval_dict.pkl"
        )
    print("Writing: " + result_dict_fn)
    with open(result_dict_fn, "wb") as f:
        pickle.dump(result_dict, f, -1)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Set options
    options_dict = default_options_dict.copy()
    options_dict["data_dir"] = path.normpath(args.data_dir)
    options_dict["wip"] = args.wip
    options_dict["rnd_seed"] = args.rnd_seed
    options_dict["min_duration"] = args.min_duration
    options_dict["init_am_n_iter"] = args.init_am_n_iter
    options_dict["segment_n_iter"] = args.segment_n_iter
    options_dict["model_dir"] = path.join(
        options_dict["model_dir"], options_dict["data_dir"].replace("data" +
        os.sep, "")
        )

    pickle_fn = ksegment(options_dict)

    if args.eval:
        eval_pickle(pickle_fn)
        print(datetime.now())


if __name__ == "__main__":
    main()
