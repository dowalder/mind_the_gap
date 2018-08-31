#!/usr/bin/env python3

"""
original version from:
https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/neural_style.py
commit: 645c7c386e62d2fb1d50f4621c1a52645a13869f
"""

import argparse
import re
import pathlib

import torch
import torch.optim
import torch.utils.data

from torchvision import transforms

import src.transformer_net
import src.file_op


def stylize(args):

    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    style_model = src.transformer_net.TransformerNet()
    state_dict = torch.load(args.model)
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
        content_imgs = \
            [src.file_op.load_image(img.as_posix(), scale=args.content_scale) for img in content_path.glob("*.jpg")]
    else:
        content_imgs = [src.file_op.load_image(content_path.as_posix(), scale=args.content_scale)]

    content_imgs = [content_transform(img).unsqueeze(0) for img in content_imgs]

    with torch.no_grad():
        for idx, img in enumerate(content_imgs):
            output = style_model(img.to(args.device)).cpu()
            path = output_path / "{}.jpg".format(idx)
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

    args = parser.parse_args()

    args.device = "cpu" if args.cuda == -1 else "cuda:{}".format(args.cuda)

    stylize(args)


if __name__ == "__main__":
    main()
