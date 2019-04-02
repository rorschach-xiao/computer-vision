import os
import re
import sys
import subprocess
import _pickle as cPickle
import numpy as np


def load_pfm(fname):
    file=open(fname,'rb')
    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode(encoding='utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.flipud(np.reshape(data, shape)), scale


def writePFM(file, image, scale=1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
    image = np.flipud(image)
    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))
    endian = image.dtype.byteorder
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
    file.write(b'%f\n' % scale)
    image.tofile(file)

#output_dir = '/Users/HK/PycharmProjects/CRL'
#assert (os.path.isdir(output_dir))
#dispnoc = []
#base1 = '/Users/HK/PycharmProjects/CRL/data/disparity/35mm_focallength/scene_forwards/fast/left'
#filenames = sorted(os.listdir(base1))
#for i in range(np.size(filenames)):
#    print(filenames[i])
#    disp, scale = load_pfm(os.path.join(base1, filenames[i]))
#    dispnoc.append(disp.astype(np.float32))
#with open(os.path.join(output_dir, 'gopro2.pkl'), 'wb') as f:
# #cPickle.dump(dispnoc, f)
# subprocess.check_call('rm -f {}/*.{{bin,dim,txt,type}}'.format(output_dir), shell=True)


# for i in range(len(dispnoc)):
#     tofile('{}/dispnoc_{:04d}.bin'.format(output_dir, i + 1), dispnoc[i])
# with open('./test.pkl', 'rb') as f:
# aaa = cPickle.load(f)