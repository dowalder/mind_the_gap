#!/usr/bin/env python3

import subprocess
import threading


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
        # ("neural_4_sib_augmented", "loop_obstacles"),
        # ("neural_4_sib_augmented", "udem1"),
        # ("neural_4_sib_augmented", "zigzag_dists"),
        # ("neural_4_sib_augmented", "real"),
        # ("neural_4_sib_augmented", "real2"),
        # ("real", "real2"),
        # ("neural_4_styles_in_batch", "real2"),
        # ("neural_style1", "real2")
        # ("20_sib", "loop_obstacles"),
        # ("20_sib", "udem1"),
        # ("20_sib", "zigzag_dists"),
        # ("20_sib", "real"),
        # ("20_sib_augmented", "loop_obstacles"),
        # ("20_sib_augmented", "udem1"),
        # ("20_sib_augmented", "zigzag_dists"),
        # ("20_sib_augmented", "real"),
        # ("4_sib_batchnorm", "loop_obstacles"),
        # ("4_sib_batchnorm", "udem1"),
        # ("4_sib_batchnorm", "zigzag_dists"),
        # ("4_sib_batchnorm", "real"),
        # ("4_sib_manyrandom", "loop_obstacles"),
        # ("4_sib_manyrandom", "udem1"),
        # ("4_sib_manyrandom", "zigzag_dists"),
        # ("4_sib_manyrandom", "real"),
        # ("4_sib_onlyroad_cropped", "loop_obstacles"),
        # ("4_sib_onlyroad_cropped", "udem1"),
        # ("4_sib_onlyroad_cropped", "zigzag_dists"),
        # ("4_sib_onlyroad_cropped", "real"),
    ]

    sets = [
        ["original/blue", "original/red", "original/green", "original/b_white", "original/m_white", "original/l_white"],
        ["20_sib_cropped/blue", "20_sib_cropped/red", "20_sib_cropped/green", "20_sib_cropped/b_white", "20_sib_cropped/m_white", "20_sib_cropped/l_white"],
        ["style3/blue", "style3/red", "style3/green", "style3/b_white", "style3/m_white", "style3/l_white"],
        ["pix2pix/blue", "pix2pix/red", "pix2pix/green", "pix2pix/b_white", "pix2pix/m_white", "pix2pix/l_white"]
    ]
    for set in sets:
        for idx_s in range(len(set)):
            for idx_t in range(idx_s + 1, len(set)):
                comparisons.append((set[idx_s], set[idx_t]))

    datapath = "/home/dwalder/dataspace/fid/"
    execuatable = "./fid_score.py"

    def get_comparison():
        if comparisons:
            return comparisons.pop()
        else:
            return None

    lock = threading.Lock()

    def compute_fid():
        while True:
            lock.acquire(blocking=True)
            try:
                comp = get_comparison()
                if comp is None:
                    break
                set1, set2 = comp
            finally:
                lock.release()
            ret = subprocess.run([execuatable, datapath + set1, datapath + set2], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, universal_newlines=True)
            msg = ""
            if ret.returncode:
                msg = "{} exited with code {}:\n{}".format(ret.args, ret.returncode, ret.stdout)

            out = str(ret.stdout).split()
            try:
                fid = float(out[-1][:4])
            except ValueError as e:
                msg = "Could not read out FID score for {} vs {}:\nout: {}\nerr: {}".format(set1, set2, ret.stdout, e)

            if not msg:
                msg = "{} vs {}: {}".format(set1, set2, fid)

            lock.acquire(blocking=True)
            try:
                print(msg)
            finally:
                lock.release()

    threads = []
    for _ in range(8):
        thread = threading.Thread(target=compute_fid)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
