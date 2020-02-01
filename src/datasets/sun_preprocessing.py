import argparse
import numpy as np
import scipy.io as sio

from os.path import join


def create_train_test_split(images_mat, classes_txt, split_txt):
    image_data = sio.loadmat(images_mat)
    image_paths = image_data['images']

    classes = set()
    img_per_cls = {}
    for img_path in image_paths:
        full_path = img_path[0][0]
        splits = full_path.split('/')
        cls = splits[1:len(splits)-1]
        cls = '_'.join(cls)
        classes.add(cls)
        if cls not in img_per_cls:
            img_per_cls[cls] = []
        img_per_cls[cls].append(full_path)

    print("Num classes: {}".format(len(classes)))
    cls_list = sorted(list(classes))

    split_data = []
    test_size = 0.2
    img_id = 0
    for cls_id, cls in enumerate(cls_list):
        imgs = img_per_cls[cls]
        cnt = len(imgs)
        train_test = np.ones(cnt)
        train_test[:int(np.round(cnt * test_size))] = 0
        train_test = np.random.permutation(train_test)
        for i, img in enumerate(imgs):
            split_data.append('{} images/{} {} {}'.format(img_id,
                                                          img,
                                                          cls_id + 1,
                                                          int(train_test[i])))
            img_id += 1

    with open(classes_txt, 'w') as f:
        for i, cls in enumerate(cls_list):
            f.write("{} {}\n".format(i, cls))

    with open(split_txt, 'w') as f:
        f.write('\n'.join(split_data))


def convert_attribute_matrix(images_mat, split_txt, raw_attributes_mat,
                             cont_attributes_mat, attributes_npy):
    img_dict = {}
    with open(split_txt) as f:
        for i, line in enumerate(f):
            _, name, cls, _ = line.split()
            img_dict[name] = int(cls) - 1
            assert img_dict[name] >= 0 and img_dict[name] < 717

    imgs = sio.loadmat(images_mat)['images']
    attr_mat = sio.loadmat(cont_attributes_mat)['labels_cv']

    attr_names = sio.loadmat(raw_attributes_mat)['attributes']
    print('Attributes:')
    for i in range(0, len(attr_names), 5):
        print(i, [n[0][0] for n in attr_names[i:i+5]])

    attributes = np.zeros((717, 102))
    attr_cnt = np.zeros(717)

    for img, attr in zip(imgs, attr_mat):
        img_name = img[0][0]
        cls = img_dict['images/'+img_name]
        attributes[cls] += attr
        attr_cnt[cls] += 1

    attributes = attributes/attr_cnt[:, np.newaxis]

    np.savetxt(attributes_npy, attributes.T)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess SUN Dataset.')
    parser.add_argument('-data_path', default='./sun', type=str,
                        help='Path to the SUN dataset.')
    parser.add_argument('-seed', default=1, type=int,
                        help='Seed for split generation.')
    args = parser.parse_args()

    np.random.seed(args.seed)

    images_mat = join(args.data_path, 'images.mat')
    classes_txt = join(args.data_path, 'classes.txt')
    split_txt = join(args.data_path, 'train_test_classification_split.txt')
    raw_attributes_mat = join(args.data_path, 'attributes.mat')
    cont_attributes_mat = join(args.data_path,
                               'attributeLabels_continuous.mat')
    attributes_npy = join(args.data_path, 'attributes_continuous.npy')

    create_train_test_split(images_mat, classes_txt, split_txt)
    convert_attribute_matrix(images_mat, split_txt, raw_attributes_mat,
                             cont_attributes_mat, attributes_npy)
