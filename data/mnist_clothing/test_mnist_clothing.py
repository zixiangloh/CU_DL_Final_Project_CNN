import numpy as np
import matplotlib.pyplot as plt
import struct


imagefile = 'train-images-idx3-ubyte'
labelfile = 'train-labels-idx1-ubyte'

with open(imagefile, 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    print("Magic: {:d}, Size: {:d}, NRows: {:d}, NCols: {:d}".format(magic, size, nrows, ncols))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))

plt.imshow(data[0, :, :], cmap=plt.cm.binary_r)
plt.show()

with open(labelfile, 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    print("Magic: {:d}, Size: {:d}".format(magic, size))
    data_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data_labels = data_labels.reshape((size, ))

print(data_labels[0])
