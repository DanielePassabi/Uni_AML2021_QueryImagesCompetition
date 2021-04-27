import os
import cv2

from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from scipy import spatial
from sklearn.preprocessing import StandardScaler

from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Challenge presentation example')
parser.add_argument('--data_path',
                    '-d',
                    type=str,
                    default='challenge_data_small',
                    help='Dataset path')
parser.add_argument('--descriptor',
                    '-desc',
                    type=str,
                    default='sift',
                    help='Descriptor to be used')
parser.add_argument('--output_dim',
                    '-o',
                    type=int,
                    default=10,
                    help='Descriptor length')
parser.add_argument('--save_dir',
                    '-s',
                    type=str,
                    default=None,
                    help='Save or not gallery/query feats')
parser.add_argument('--gray',
                    '-g',
                    action='store_true',
                    help='Grayscale/RGB SIFT')
parser.add_argument('--random',
                    '-r',
                    action='store_true',
                    help='Random run')
args = parser.parse_args()


class Dataset(object):
    def __init__(self, data_path):
        self.data_path = data_path
        assert os.path.exists(self.data_path), 'Insert a valid path!'

        self.data_classes = os.listdir(self.data_path)

        self.data_mapping = {}

        for c, c_name in enumerate(self.data_classes):
            temp_path = os.path.join(self.data_path, c_name)
            temp_images = os.listdir(temp_path)

            for i in temp_images:
                img_tmp = os.path.join(temp_path, i)

                if img_tmp.endswith('.jpg'):
                    if c_name == 'distractor':
                        self.data_mapping[img_tmp] = -1
                    else:
                        self.data_mapping[img_tmp] = int(c_name)

        print('Loaded {:d} from {:s} images'.format(len(self.data_mapping.keys()),
                                                    self.data_path))

    def get_data_paths(self):
        images = []
        classes = []
        for img_path in self.data_mapping.keys():
            if img_path.endswith('.jpg'):
                images.append(img_path)
                classes.append(self.data_mapping[img_path])
        return images, np.array(classes)


    def num_classes(self):
        return len(self.data_classes)


class Histogram:
    def __init__(self, bins):
        self.bins = bins

    def detectAndCompute(self, image, other):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist(images=[image], channels=[0, 1, 2], mask=None,
                            histSize=self.bins, ranges=[0, 256] * 3)
        hist = cv2.normalize(hist, dst=hist.shape).flatten()
        hist = hist[np.newaxis, :]
        return None, hist


class FeatureExtractor(object):

    def __init__(self, feature_extractor, gray=False):

        self.feature_extractor = feature_extractor
        self.gray = gray

    def get_descriptor(self, img_path):
        img = cv2.imread(img_path)
        if self.gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, descs = self.feature_extractor.detectAndCompute(img, None)
        return descs


    def extract_features(self, data_list):
        # we init features
        features = []

        for i, img_path in enumerate(tqdm(data_list, desc='Extraction')):
            # get descriptor
            histo = self.get_descriptor(img_path)
            features.append(histo)
        features = np.concatenate(features)

        return features


def topk_accuracy(gt_label, matched_label, k=1):
    matched_label = matched_label[:, :k]
    total = matched_label.shape[0]
    correct = 0
    for q_idx, q_lbl in enumerate(gt_label):
        correct+= np.any(q_lbl == matched_label[q_idx, :]).item()
    acc_tmp = correct/total

    return acc_tmp


def main():

    # we define training dataset
    training_path = os.path.join(args.data_path, 'training')

    # we define validation dataset
    validation_path = os.path.join(args.data_path, 'validation')
    gallery_path = os.path.join(validation_path, 'gallery')
    query_path = os.path.join(validation_path, 'query')

    training_dataset = Dataset(data_path=training_path)
    gallery_dataset = Dataset(data_path=gallery_path)
    query_dataset = Dataset(data_path=query_path)

    # get training data and classes
    training_paths, _ = training_dataset.get_data_paths()

    # we get validation gallery and query data

    gallery_paths, gallery_classes = gallery_dataset.get_data_paths()
    query_paths, query_classes = query_dataset.get_data_paths()

    if not args.random:
        
        feature_extractor = Histogram(bins=[32, 32, 32])

        # we define the feature extractor providing the model
        extractor = FeatureExtractor(feature_extractor=feature_extractor, gray=args.gray)

        # now we can use features
        # we get query features
        query_features = extractor.extract_features(query_paths)

        # we get gallery features
        gallery_features = extractor.extract_features(gallery_paths)

        print(gallery_features.shape, query_features.shape)
        

        pairwise_dist = spatial.distance.cdist(query_features, gallery_features, 'minkowski', p=2.)

        print('--> Computed distances and got c-dist {}'.format(pairwise_dist.shape))

        indices = np.argsort(pairwise_dist, axis=-1)

    else:
        indices = np.random.randint(len(gallery_paths),
                                    size=(len(query_paths),len(gallery_paths)))
    

    gallery_matches = gallery_classes[indices]
    
    print('########## RESULTS ##########')

    for k in [1, 3, 10]:
        topk_acc = topk_accuracy(query_classes, gallery_matches, k)
        print('--> Top-{:d} Accuracy: {:.3f}'.format(k, topk_acc))
    


if __name__ == '__main__':
    main()
