#!/usr/bin/env python

"""
Extract syllable boundary landmarks.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015, 2021
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import numpy as np
import pickle
import sys

sys.path.append("..")
sys.path.append(str(Path("..")/".."/"src"/"eskmeans"/"utils"))

from paths import buckeye_datadir, xitsonga_datadir
import theta_oscillator


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("dataset", type=str, choices=["buckeye", "xitsonga"])
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Get list of utterance segments
    if args.dataset == "buckeye":
        split_keys = {}
        utterance_keys = []
        for split in ["devpart1", "zs"]:
            features_fn = (
                (Path("..")/"features"/"mfcc"/args.dataset/split).with_suffix(
                ".dd.npz")
                )           
            features = np.load(features_fn)
            utterance_keys.extend(list(features))
            if not split in split_keys:
                split_keys[split] = []
            split_keys[split].extend(list(features))
    elif args.dataset == "xitsonga":
        features_fn = (
            (Path("..")/"features"/"mfcc"/args.dataset/"xitsonga").with_suffix(
            ".dd.npz")
            )
        features = np.load(features_fn)
        utterance_keys = list(features)
    utterance_keys = sorted(utterance_keys)
    print("No. utterances: {}".format(len(utterance_keys)))

    # Get all audio for this dataset
    wav_dict = {}
    if args.dataset == "buckeye":
        for wav_fn in tqdm(Path(buckeye_datadir).glob("*/*.wav")):
            signal, sample_rate = librosa.core.load(wav_fn, sr=None)
            basename = wav_fn.stem
            wav_dict["{}_{}".format(basename[:3], basename[3:6])] = signal
    elif args.dataset == "xitsonga":
        for wav_fn in tqdm(Path(xitsonga_datadir).glob("*.wav")):
            signal, sample_rate = librosa.core.load(wav_fn, sr=None)
            basename = wav_fn.stem
            corpus, lang, speaker, utt = basename.split("_")
            wav_dict["{}_{}-{}-{}".format(speaker, corpus, lang, utt)] = signal

    # For each utterance segment
    executor = ProcessPoolExecutor(max_workers=cpu_count())
    futures = []
    for utt_key in tqdm(utterance_keys):
    # for utt_key in ["s02_01a_000271-000290",]:

        # Get start and end samples
        speaker, utt, start_end = utt_key.split("_")
        start, end = start_end.split("-")
        start = float(start)
        end = float(end)
        start_sample = int(start/100*sample_rate)
        end_sample = int(end/100*sample_rate)

        # Extract utterance from larger wav
        audio = wav_dict["{}_{}".format(speaker, utt)][start_sample:end_sample]

        # Syllable boundary detection: Add to parallel stream
        futures.append(executor.submit(
            theta_oscillator.get_boundaries, audio, fs=sample_rate
            ))

        # Syllable boundary detection
        # boundaries = theta_oscillator.get_boundaries(audio, fs=sample_rate)

    # Store landmarks
    results = [future.result() for future in tqdm(futures)]
    landmarks_dict = {}
    for i_utt, utt_key in enumerate(utterance_keys):
        boundaries = results[i_utt]
        landmarks_dict[utt_key] = list(
            np.asarray(np.ceil(boundaries*100), dtype=np.int32)
            )[1:]  # remove first (0) landmark

    # Write landmarks
    if args.dataset == "buckeye":
        # If this is the Buckeye dataset, split into sets
        split_landmarks = {}
        for split in ["devpart1", "zs"]:
            for utt_key in split_keys[split]:
                split_landmarks[utt_key] = landmarks_dict[utt_key]
            
            output_dir = Path("landmarks")/split
            output_dir.mkdir(parents=True, exist_ok=True)
            landmarks_fn = output_dir/"landmarks.unsup_syl.pkl"
            print("Writing: {}".format(landmarks_fn))
            with open(landmarks_fn, "wb") as f:
                pickle.dump(split_landmarks, f)
    elif args.dataset == "xitsonga":
        output_dir = Path("landmarks")/args.dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        landmarks_fn = output_dir/"landmarks.unsup_syl.pkl"
        print("Writing: {}".format(landmarks_fn))
        with open(landmarks_fn, "wb") as f:
            pickle.dump(landmarks_dict, f)


if __name__ == "__main__":
    main()
