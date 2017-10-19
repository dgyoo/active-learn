import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def make_dataset_train(root):
    root_train = os.path.join(root, 'train')
    classes = [d for d in os.listdir(root_train) if os.path.isdir(os.path.join(root_train, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    pairs = []
    root_train = os.path.expanduser(root_train)
    for target in sorted(os.listdir(root_train)):
        d = os.path.join(root_train, target)
        if not os.path.isdir(d): continue
        for rootdir, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if any(fname.endswith(extension) for extension in IMG_EXTENSIONS):
                    path = os.path.join(rootdir, fname)
                    pairs.append((path, class_to_idx[target]))
    return {'pairs': pairs, 'classes': classes}

def make_dataset_val(root):
    root_train = os.path.join(root, 'train')
    root_val = os.path.join(root, 'val')
    classes = [d for d in os.listdir(root_train) if os.path.isdir(os.path.join(root_train, d))]
    classes.sort()
    pairs = []
    with open(os.path.join(root_val, 'val.txt'), 'r') as fp:
        for line in fp:
            fname, class_idx = line.rstrip('\n').split(' ')
            fpath = os.path.join(root_val, fname)
            class_idx = int(class_idx)
            assert os.path.isfile(fpath), 'No such file: {}'.format(fpath)
            pairs.append((fpath, class_idx))
    return {'pairs': pairs, 'classes': classes}
