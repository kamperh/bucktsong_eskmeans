#!/usr/bin/env python

"""
Perform dense downsampling over indicated segmentation intervals.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015, 2021
"""

from datetime import datetime
from os import path
from tqdm import tqdm
import argparse
import numpy as np
import pickle
import scipy.signal as signal
import sys

output_dir = "output"


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("subset", type=str, choices=["devpart1", "zs", "xitsonga"], help="target subset")
    parser.add_argument("landmarks", type=str, choices=["gtphone", "unsup_syl"], help="landmarks set")
    parser.add_argument(
        "feature_type", type=str, help="input feature type", choices=["mfcc"]
        )
    parser.add_argument("--n", type=int, help="number of samples (default: %(default)s)", default=10)
    parser.add_argument(
        "--frame_dims", type=int, default=None,
        help="only keep these number of dimensions"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def downsample_utterance(features, seglist, n):
    """
    Return the downsampled matrix with each row an embedding for a segment in
    the seglist.
    """
    embeddings = []
    for i, j in seglist:
        y = features[i:j+1, :].T
        y_new = signal.resample(y, n, axis=1).flatten("C")
        embeddings.append(y_new)
    return np.asarray(embeddings)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    if args.subset == "xitsonga":
        input_npz_fn = path.join(
            "..", "features", args.feature_type, "xitsonga", args.subset +
            ".dd.npz"
            )
    else:
        input_npz_fn = path.join(
            "..", "features", args.feature_type, "buckeye", args.subset +
            ".dd.npz"
            )

    print("Reading:", input_npz_fn)
    input_npz = np.load(input_npz_fn)
    d_frame = input_npz[list(input_npz)[0]].shape[1]
    print("No. of utterances:", len(list(input_npz)))

    seglist_pickle_fn = path.join(output_dir, args.subset, "seglist." + args.landmarks + ".pkl")
    print("Reading:", seglist_pickle_fn)
    with open(seglist_pickle_fn, "rb") as f:
        seglist_dict = pickle.load(f)
    print("No. of utterances:", len(seglist_dict))

    print("Frame dimensionality:", d_frame)
    if args.frame_dims is not None and args.frame_dims < d_frame:
        d_frame = args.frame_dims
        print("Reducing frame dimensionality:", d_frame)

    print("No. of samples:", args.n)

    # Temp
    # print(list(seglist_dict)[0])
    # print(list(input_npz)[0])
    # print(len(list(seglist_dict)))
    # print(len(list(input_npz)))
    # print(sorted(list(seglist_dict))[24400])
    # print(sorted(list(input_npz))[24400])
    # assert False
    # print(seglist_dict[sorted(list(seglist_dict))[2991]])
    # assert False
    # print("s02_01a_004199-004219" in seglist_dict)
    # assert False

    # # Temp
    # landmarks_dict = pickle.load(open("output/devpart1/landmarks.unsup_syl.pkl", "rb"))
    # # print("s02_01a_004199-004220" in landmarks_dict)
    # # assert False
    # new_landmarks_dict = {}
    # new_seglist_dict = {}
    # for i, utterance in enumerate(sorted(input_npz)):
    #     if utterance not in seglist_dict:
    #         speaker, utt, start_end = utterance.split("_")
    #         start, end = start_end.split("-")
    #         start = int(start)
    #         end = int(end)
    #         solved = 0
    #         for seglist_utterance in seglist_dict:
    #             if not seglist_utterance.startswith(speaker + "_" + utt):
    #                 continue
    #             seglist_speaker, seglist_utt, seglist_start_end = seglist_utterance.split("_")
    #             seglist_start, seglist_end = seglist_start_end.split("-")
    #             seglist_start = int(seglist_start)
    #             seglist_end = int(seglist_end)
    #             if ((seglist_start == start) or (seglist_end == end) or (abs(seglist_end - end) <= 1)):
    #                 new_seglist_dict[utterance] = seglist_dict[seglist_utterance]
    #                 new_landmarks_dict[utterance] = landmarks_dict[seglist_utterance]
    #                 # print(seglist_start, start, "!!!")
    #                 print(utterance, seglist_utterance)
    #                 solved += 1
    #                 # break
    #         assert solved == 1
    #     else:
    #         new_seglist_dict[utterance] = seglist_dict[utterance]
    #         new_landmarks_dict[utterance] = landmarks_dict[utterance]
    # seglist_dict = new_seglist_dict
    # with open("seglist.tmp.pkl", "wb") as f:
    #     pickle.dump(seglist_dict, f)
    # with open("landmarks.tmp.pkl", "wb") as f:
    #     pickle.dump(new_landmarks_dict, f)
    # assert False

    print(datetime.now())
    print("Downsampling:")
    downsample_dict = {}
    for utt in tqdm(input_npz):
        downsample_dict[utt] = downsample_utterance(
            input_npz[utt][:, :args.frame_dims], seglist_dict[utt], args.n
            )
    print(datetime.now())

    output_npz_fn = path.join(
        output_dir, args.subset, "downsample_dense." + args.feature_type +
        ".n_" + str(args.n) + "." + args.landmarks + ".npz"
        )
    print("Writing:", output_npz_fn)
    np.savez_compressed(output_npz_fn, **downsample_dict)


if __name__ == "__main__":
    main()
