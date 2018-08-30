"""
original version from:
https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/neural_style.py
commit: 645c7c386e62d2fb1d50f4621c1a52645a13869f
"""

import argparse
import os
import sys
import time
import re
import pathlib

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision

import src.vgg
import src.transformer_net
import src.file_op
import src.utils


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.CenterCrop(args.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = torchvision.datasets.ImageFolder(args.dataset, transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    transformer = src.transformer_net.TransformerNet().to(args.device)
    optimizer = torch.optim.Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = src.vgg.Vgg16(requires_grad=False).to(args.device)
    style_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.mul(255))
    ])
    if args.use_multiple_styles:
        path = pathlib.Path(args.style_image)
        if not path.is_dir():
            raise ValueError("If --use-multiple-styles is specified, --style-image must be a folder containing "
                             "multiple style images")
        styles = [src.file_op.load_image(img, size=args.style_size) for img in path.glob("*.jpg")]
        styles = [style_transform(style) for style in styles]
        if len(styles) < args.batch_size:
            raise ValueError("Found only {} images in {}, but need at least {}".format(
                len(styles), path, args.batch_size))
        style = torch.empty((args.batch_size, 3, args.style_size, args.style_size), device=args.device,
                            dtype=torch.float)
        for idx, st in enumerate(styles):
            style[idx, :, :, :] = style

    else:
        style = src.file_op.load_image(args.style_image, size=args.style_size)
        style = style_transform(style)
        style = style.repeat(args.batch_size, 1, 1, 1).to(args.device)

    features_style = vgg(src.utils.normalize_batch(style))

    gram_style = [src.utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(args.device)
            y = transformer(x)

            y = src.utils.normalize_batch(y)
            x = src.utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = src.utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                    agg_content_loss / args.log_interval,
                    agg_style_loss / args.log_interval,
                    (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                agg_content_loss = 0
                agg_style_loss = 0
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(args.device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def main():
    parser = argparse.ArgumentParser(description="parser for fast-neural-style")

    parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    parser.add_argument("--cuda", type=int, required=True,
                                  help="index of gpu to use, -1 is cpu")
    parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    parser.add_argument("--use-multiple-styles", action="store_true",
                        help="training from multiple style image, not only one")

    args = parser.parse_args()

    args.device = "cpu" if args.cuda < 0 else "cuda:{}".format(args.cuda)

    train(args)


if __name__ == "__main__":
    main()
