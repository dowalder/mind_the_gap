#!/usr/bin/env python3

"""
original version from:
https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/neural_style.py
commit: 645c7c386e62d2fb1d50f4621c1a52645a13869f
"""

import argparse
import re
import pathlib
import time

import torch
import torch.optim
import torch.utils.data

from torchvision import transforms

import src.transformer_net
import src.file_op


def stylize(args):

    content_transform = [transforms.Grayscale()] if args.grayscale else []
    content_transform += [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ]
    content_transform = transforms.Compose(content_transform)

    style_model = src.transformer_net.TransformerNet(
        norm_layer=src.transformer_net.get_norm_layer(args.norm),
        input_channels=(1 if args.grayscale else 3),
        coord_conv=args.coord_conv
    )
    state_dict = torch.load(args.wheights)
    if args.norm == "instance":
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
    style_model.load_state_dict(state_dict)
    style_model.to(args.device)

    src_folder = pathlib.Path(args.src_dir)
    tgt_folder = pathlib.Path(args.tgt_dir)
    assert tgt_folder.is_dir()
    assert src_folder.is_dir()

    def transform_folder(src_dir: pathlib.Path, tgt_dir: pathlib.Path):
        t = time.time()
        for img_path in src_dir.iterdir():
            if img_path.is_dir() and args.recursive:
                tgt = tgt_dir / img_path.name
                tgt.mkdir(exist_ok=True)
                transform_folder(img_path, tgt)
            if img_path.suffix not in [".jpg", ".png"]:
                continue
            img = src.file_op.load_image(img_path.as_posix(), scale=args.content_scale)
            with torch.no_grad():
                img = content_transform(img).unsqueeze(0).to(args.device)
                output = style_model(img).cpu()

            tgt_path = tgt_dir / img_path.name
            src.file_op.save_image(tgt_path.as_posix(), output[0])
        print("Finished {} in {:4.2f}s".format(src_dir, time.time() - t))

    transform_folder(src_folder, tgt_folder)


def main():
    parser = argparse.ArgumentParser(description="parser for fast-neural-style")

    parser.add_argument("--src_dir", type=str, required=True, help="path to content image you want to stylize")
    parser.add_argument("--content-scale", type=float, default=None, help="factor for scaling down the content image")
    parser.add_argument("--tgt_dir", type=str, required=True, help="path for saving the output images")
    parser.add_argument("--wheights", type=str, required=True, help="saved wheights to be used for stylizing the image")
    parser.add_argument("--cuda", default=0, help="number of GPU to use. -1 for CPU.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--norm", choices=["batch", "instance", "none"], default="instance")
    parser.add_argument("--grayscale", action="store_true", help="convert the image to grayscale before transformation")
    parser.add_argument("--coord_conv", action="store_true", help="use coord conv layers instead of normal conv")
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--recursive", action="store_true")

    args = parser.parse_args()

    args.device = "cpu" if args.cuda == -1 else "cuda:{}".format(args.cuda)

    stylize(args)


if __name__ == "__main__":
    main()
