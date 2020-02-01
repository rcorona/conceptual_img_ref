import argparse
import numpy as np
import os

from os.path import join


def create_train_test_split(images_dir, classes_txt, split_txt):
    classes = {}
    class_cnts = {}
    with open(classes_txt, 'r') as f:
        for line in f:
            cid, cls = line.split()
            classes[cls] = int(cid)
            class_cnts[cls] = 0

    data_str = []
    i = 0
    for root, dirs, files in os.walk(images_dir):
        dirs.sort()
        files.sort()
        for f in files:
            if f.endswith('.jpg'):
                root_split = root.split('/')
                cls = root_split[-1]
                file_path = join(root_split[-2], root_split[-1], f)
                data_str.append('{} {} {}'.format(i,
                                                  file_path,
                                                  classes[cls]))
                i += 1
                class_cnts[cls] += 1

    test_size = 0.2
    offset = 0
    for cls, cnt in sorted(class_cnts.items()):
        train_test = np.ones(cnt)
        train_test[:int(np.round(cnt * test_size))] = 0
        train_test = np.random.permutation(train_test)

        for i in range(cnt):
            data_str[offset + i] += ' {}'.format(int(train_test[i]))

        offset += cnt

    with open(split_txt, 'w') as f:
        f.write('\n'.join(data_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess AWA2 Dataset.')
    parser.add_argument('-data_path', default='./awa', type=str,
                        help='Path to the AWA2 dataset.')
    parser.add_argument('-seed', default=1, type=int,
                        help='Seed for split generation.')
    args = parser.parse_args()

    np.random.seed(args.seed)

    images_mat = join(args.data_path, 'JPEGImages')
    classes_txt = join(args.data_path, 'classes.txt')
    split_txt = join(args.data_path, 'train_test_classification_split.txt')

    create_train_test_split(images_mat, classes_txt, split_txt)
