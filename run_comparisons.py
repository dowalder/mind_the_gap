#!/usr/bin/env python3

import subprocess


def main():
    comparisons = [
        ("udem1", "zigzag_dists"),
        ("udem1", "loop_obstacles"),
        ("loop_obstacles", "zigzag_dists"),
        ("neural_style1", "loop_obstacles"),
        ("neural_style1", "zigzag_dists"),
        ("neural_style1", "udem1"),
        ("real", "loop_obstacles"),
        ("real", "zigzag_dists"),
        ("real", "udem1"),
        ("randbackgradscale", "loop_obstacles_onlymarks"),
        ("randbackgradscale", "zigzag_dists_onlymarks"),
        ("randbackgradscale", "udem1_onlymarks"),

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
