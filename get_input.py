import scipy.misc as misc


def get_feedict(values, keys):
  dic = {}
  for i in range(len(values)):
    dic[keys[i]] = values[i]
  return dic


def get_im(im_dir, num):
  im_array = misc.imread(im_dir[num])
  im_m, im_g = im_array[:, 0:512, :], im_array[:, 512:, :]
  return im_m[None, :, :, :], im_g[None, :, :, :]
