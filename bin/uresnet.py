import os
import sys

URESNET_DIR = os.path.dirname(os.path.abspath(__file__))
URESNET_DIR = os.path.dirname(URESNET_DIR)
sys.path.insert(0, URESNET_DIR)
from uresnet.flags import URESNET_FLAGS
from ROOT import TChain
from larcv import larcv
import numpy as np


def main():
    print('I am in pyhton got %s %s' % (2, 4))
    print(os.path.dirname(os.path.abspath(__file__)))
    sys.argv = ['uresnet.py', 'inference', '--full', '-pl', '1', '-uf', '16', '-mp', 'Plane0snapshot-13999.ckpt', '-io',
                'larcv_sparse', '-bs', '1', '-nc', '5', '-rs', '1', '-ss', '512', '-dd', '2', '-uns', '5', '-dkeys',
                'wire,label', '-mn', 'uresnet_sparse', '-it', '1', '-ld', 'log', '-if', 'larcv_5label.root'];
    print(len(sys.argv))
    #    for x in sys.argv:
    #        print(x)

    print ('Before calling flags print %s' % 2)
    #    flags = URESNET_FLAGS()
    #    flags.RanTest(2,4)

    #    flags = URESNET_FLAGS()
    #    flags.parse_args()

    br_blob = {}
    blob = {}

    ch = TChain('sparse2d_wire_tree')
    ch.AddFile('larcv_5label.root')  # Adding the files RanItay
    ch.GetEntry(0)
    br_blob['wire'] = getattr(ch, 'sparse2d_wire_branch')
    br_data = br_blob['wire']
    br_data = br_data.as_vector().front()
    num_point = br_blob['wire'].as_vector().front().as_vector().size()
    np_voxel = np.zeros(shape=(num_point, 2), dtype=np.int32)
    as_numpy_voxels = larcv.fill_2d_voxels
    as_numpy_pcloud = larcv.fill_2d_pcloud
    as_numpy_voxels(br_data, np_voxel)
    blob['voxels'] = np_voxel
    np_feature = np.zeros(shape=(num_point, 1), dtype=np.float32)
    as_numpy_pcloud(br_data, np_feature)
    blob['feature'] = np_feature
    blob['batch_id'] = np.zeros(shape=(num_point, 1), dtype=np.int32)

    data = np.hstack((blob['voxels'], blob['batch_id'], blob['feature']))

    print('bla')


#


if __name__ == '__main__':
    main()
