#!/usr/bin/env python3

import subprocess


def main():
    comparisons = [
        # ("udem1", "zigzag_dists"),
        # ("udem1", "loop_obstacles"),
        # ("loop_obstacles", "zigzag_dists"),
        # ("neural_style1", "loop_obstacles"),
        # ("neural_style1", "zigzag_dists"),
        # ("neural_style1", "udem1"),
        # ("real", "loop_obstacles"),
        # ("real", "zigzag_dists"),
        # ("real", "udem1"),
        # ("randbackgradscale", "loop_obstacles_onlymarks"),
        # ("randbackgradscale", "zigzag_dists_onlymarks"),
        # ("randbackgradscale", "udem1_onlymarks"),
        # ("neural_4_styles_in_batch", "udem1"),
        # ("neural_4_styles_in_batch", "loop_obstacles"),
        # ("neural_4_styles_in_batch", "zigzag_dists"),
        # ("neural_4_styles_in_batch", "real"),
        # ("neural_style1", "real"),
        # ("neural_4_styles_in_batch_cropped", "loop_obstacles"),
        # ("neural_4_styles_in_batch_cropped", "udem1"),
        # ("neural_4_styles_in_batch_cropped", "zigzag_dists"),
        # ("neural_4_styles_in_batch_cropped", "real"),
        # ("neural_4_styles_in_batch_onlyroad", "loop_obstacles"),
        # ("neural_4_styles_in_batch_onlyroad", "udem1"),
        # ("neural_4_styles_in_batch_onlyroad", "zigzag_dists"),
        # ("neural_4_styles_in_batch_onlyroad", "real"),
        # ("neural_style1", "real"),
        # ("neural_4_sib_overflowed", "loop_obstacles"),
        # ("neural_4_sib_overflowed", "udem1"),
        # ("neural_4_sib_overflowed", "zigzag_dists"),
        # ("neural_4_sib_overflowed", "real"),
        # ("neural_4_sib_inverted", "loop_obstacles"),
        # ("neural_4_sib_inverted", "udem1"),
        # ("neural_4_sib_inverted", "zigzag_dists"),
        # ("neural_4_sib_inverted", "real"),
        ("neural_4_sib_augmented", "loop_obstacles"),
        ("neural_4_sib_augmented", "udem1"),
        ("neural_4_sib_augmented", "zigzag_dists"),
        ("neural_4_sib_augmented", "real"),
        ("neural_4_sib_augmented", "real2"),
        ("real", "real2"),
        ("neural_4_styles_in_batch", "real2"),
        ("neural_style1", "real2")
    ]

    datapath = "/home/dwalder/dataspace/fid/"
    execuatable = "./fid_score.py"

    for set1, set2 in comparisons:
        ret = subprocess.run([execuatable, datapath + set1, datapath + set2], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True)
        if ret.returncode:
            print("{} exited with code {}:\n{}".format(ret.args, ret.returncode, ret.stdout))
            continue
        out = str(ret.stdout).split()
        try:
            fid = float(out[-1][:4])
        except ValueError as e:
            print("Could not read out FID score for {} vs {}:\nout: {}\nerr: {}".format(set1, set2, ret.stdout, e))
            continue

        print("{} vs {}: {}".format(set1, set2, fid))


if __name__ == "__main__":
    main()
