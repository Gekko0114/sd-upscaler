import argparse

import torch
import yaml

from model import StableDiffusionUpscaler


def main():
    parser = argparse.ArgumentParser(
        description="args for stable diffusion upscaler")
    parser.add_argument('--seed', type=int, help='seed value')
    parser.add_argument('--input_image', type=str, help='input image')
    parser.add_argument('--prompt', type=str, help='prompt')
    args = parser.parse_args()
    with open('./src/config.yaml') as file:
        config = yaml.safe_load(file.read())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.input_image is not None:
        config["input_image"] = args.input_image
    if args.prompt is not None:
        config["prompt"] = args.prompt

    cpu = torch.device("cpu")
    device = torch.device("cuda")
    sdu = StableDiffusionUpscaler(config=config, cpu=cpu, device=device)
    sdu.run()


if __name__ == '__main__':
    main()
