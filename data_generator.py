import os
from urllib.request import urlretrieve

import tensorflow as tf

from utils import tqdm_hook

_IMAGENET_SMALL_ROOT_URL = "http://image-net.org/small/"
_IMAGENET_SMALL_URLS = ["train_32x32.tar", "valid_32x32.tar"]
_IMAGENET_SMALL_TRAIN_PREFIX = "train_32x32"
_IMAGENET_SMALL_EVAL_PREFIX = "valid_32x32"
_IMAGENET_SMALL_IMAGE_SIZE = 32

_IMAGENET_MEDIUM_ROOT_URL = "http://image-net.org/small/"
_IMAGENET_MEDIUM_URLS = ["train_64x64.tar", "valid_64x64.tar"]
_IMAGENET_MEDIUM_TRAIN_PREFIX = "train_64x64"
_IMAGENET_MEDIUM_EVAL_PREFIX = "valid_64x64"
_IMAGENET_MEDIUM_IMAGE_SIZE = 64


def download_image_net(directory, filename, uri):
  """Download filename from uri unless it's already in directory.

  Copies a remote file to local if that local file does not already exist.  If
  the local file pre-exists this function call, it does not check that the local
  file is a copy of the remote.

  Remote filenames can be filepaths, any URI readable by tensorflow.gfile, or a
  URL.

  Args:
    directory: path to the directory that will be used.
    filename: name of the file to download to (do nothing if it already exists).
    uri: URI to copy (or download) from.

  Returns:
    The path to the downloaded file.
  """
  tf.gfile.MakeDirs(directory)
  filepath = os.path.join(directory, filename)
  if tf.gfile.Exists(filepath):
    tf.logging.info("Not downloading, file already found: %s" % filepath)
    return filepath

  tf.logging.info("Downloading %s to %s" % (uri, filepath))
  try:
    tf.gfile.Copy(uri, filepath)
  except tf.errors.UnimplementedError:
    if uri.startswith("http"):
      inprogress_filepath = filepath + ".incomplete"

      eg_file = uri.replace('/', ' ').split()[-1]
      with tqdm_hook.TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024,
                              miniters=1, desc=eg_file) as t:
        urlretrieve(uri,
                    filename=inprogress_filepath,
                    reporthook=t.update_to,
                    data=None)

      print()
      tf.gfile.Rename(inprogress_filepath, filepath)
    else:
      raise ValueError("Unrecognized URI: " + filepath)
  statinfo = os.stat(filepath)
  tf.logging.info("Successfully downloaded %s, %s bytes." %
                  (filename, statinfo.st_size))
  return filepath


# download_image_net("/home/tomek/inpainint/test", "train_32x32",
#                    "http://image-net.org/small/train_32x32.tar")
