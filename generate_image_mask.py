import enum
import os
from argparse import ArgumentParser

import tensorflow as tf

import create_mask_image

tf.logging.set_verbosity(tf.logging.INFO)

logger = tf.logging

home = os.path.expanduser("~")


class TrainingPaths(enum.Enum):
  MASK = 0,
  ORIGINAL_IMAGE = 1,
  MASKED_IMAGE = 2


PATHS = {
  TrainingPaths.MASK: os.path.join(home, "inpainting/masks/"),
  TrainingPaths.ORIGINAL_IMAGE: os.path.join(home, "inpainting/original-images/"),
  TrainingPaths.MASKED_IMAGE: os.path.join(home, "inpainting/masked-images/")
}


def maybe_create_paths(paths):
  for path in paths:
    tf.gfile.MakeDirs(path)
    logger.info("Created {} path".format(path))


# def create_default_path(path):
#   path = os.path.join(home, path)
#
#   return path


def build_parser():
  parser = ArgumentParser()
  parser.add_argument('--num_mask', type=int,
                      dest='num_mask', help='how many mask to generate',
                      metavar='Number of mask', required=True)

  parser.add_argument('--min_units', type=int,
                      dest='min_units', help='min units to generate',
                      metavar='Min units to generate', required=True)

  parser.add_argument('--max_units', type=int,
                      dest='max_units', help='max units to generate',
                      metavar='Max units to generate', required=True)

  parser.add_argument('--masks_path', type=str,
                      dest='masks_path', help='path to save masks',
                      metavar='Path to save masks',
                      default=PATHS[TrainingPaths.MASK])

  parser.add_argument('--original_images_path', type=str,
                      dest='original_images_path', help='path to raw image',
                      metavar='Path to raw image',
                      default=PATHS[TrainingPaths.ORIGINAL_IMAGE])

  parser.add_argument('--masked_images_path', type=str,
                      dest='masked_images_path', help='image to train',
                      metavar='Train',
                      default=PATHS[TrainingPaths.MASKED_IMAGE])
  return parser


def main():
  parser = build_parser()
  arguments = parser.parse_args()
  paths = [arguments.masks_path, arguments.original_images_path, arguments.masked_images_path]
  maybe_create_paths(paths)
  create_mask_image.save_mask(arguments.num_mask, arguments.min_units, arguments.max_units,
                              arguments.masks_path, arguments.original_images_path, arguments.masked_images_path)


if __name__ == '__main__':
  main()
