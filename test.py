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
    state_dict = torch.load(args.model)
    if args.norm == "instance":
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
    style_model.load_state_dict(state_dict)
    style_model.to(args.device)

    content_path = pathlib.Path(args.content_image)
    output_path = pathlib.Path(args.output_image)
    assert output_path.is_dir(), "--output-image must be a directory"

    if content_path.is_dir():
        content_imgs = []

        def add_images(path: pathlib.Path):
            for file in path.iterdir():
                if file.is_dir() and args.recursive:
                    add_images(file)
                elif file.suffix == ".jpg":
                    content_imgs.append(file)
        add_images(content_path)
    else:
        content_imgs = [content_path]

    with torch.no_grad():
        for idx, img_file in enumerate(content_imgs):
            img = src.file_op.load_image(img_file.as_posix(), scale=args.content_scale)
            t = time.time()
            img = content_transform(img).unsqueeze(0).to(args.device)
            if args.verbose:
                print("took {:4.4f}s".format(time.time() - t))
            output = style_model(img).cpu()
            output_dir = output_path / img_file.parent.relative_to(content_path)
            output_dir.mkdir(exist_ok=True, parents=True)
            path = output_dir / img_file.name
            src.file_op.save_image(path.as_posix(), output[0])


def main():
    parser = argparse.ArgumentParser(description="parser for fast-neural-style")

    parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
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
