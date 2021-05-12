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
                    default='dataset',
                    help='Dataset path')
parser.add_argument('--output_dim',
                    '-o',
                    type=int,
                    default=20,
                    help='Descriptor length')
parser.add_argument('--save_dir',
                    '-s',
                    type=str,
                    default=None,
                    help='Save or not gallery/query feats')
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


class FeatureExtractor(object):

    def __init__(self, feature_extractor, model, out_dim=20, scale=None,
                 subsample=100):

        self.feature_extractor = feature_extractor
        self.model = model
        self.scale = scale
        self.subsample = subsample

    def get_descriptor(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, descs = self.feature_extractor.detectAndCompute(img, None)
        return descs


    def fit_model(self, data_list):
        training_feats = []
        # we extact SIFT descriptors
        for img_path in tqdm(data_list, desc='Fit extraction'):
            descs = self.get_descriptor(img_path)
            
            if descs is None:
                continue
            
            if self.subsample:
                # TODO: change here
                sub_idx = np.random.choice(np.arange(descs.shape[0]), self.subsample)
                descs = descs[sub_idx, :]

            training_feats.append(descs)
        training_feats = np.concatenate(training_feats)
        print('--> Model trained on {} features'.format(training_feats.shape))
        # we fit the model
        self.model.fit(training_feats)
        print('--> Model fitted')


    def fit_scaler(self, data_list):
        features = self.extract_features(data_list)
        print('--> Scale trained on {}'.format(features.shape))
        self.scale.fit(features)
        print('--> Scale fitted')


    def extract_features(self, data_list):
        # we init features
        features = np.zeros((len(data_list), self.model.n_clusters))

        for i, img_path in enumerate(tqdm(data_list, desc='Extraction')):
            # get descriptor
            descs = self.get_descriptor(img_path)
            # 2220x128 descs
            preds = self.model.predict(descs)
            histo, _ = np.histogram(preds, bins=np.arange(self.model.n_clusters+1), density=True)
            # append histogram
            features[i, :] = histo

        return features


    def scale_features(self, features):
        # we return the normalized features
        return self.scale.transform(features)


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
    training_paths, training_classes = training_dataset.get_data_paths()

    # we get validation gallery and query data

    gallery_paths, gallery_classes = gallery_dataset.get_data_paths()
    query_paths, query_classes = query_dataset.get_data_paths()

    if not args.random:
        

        feature_extractor = cv2.SIFT_create()

        # we define model for clustering
        model = KMeans(n_clusters=args.output_dim, n_init=10, max_iter=5000, verbose=False)
        # model = MiniBatchKMeans(n_clusters=args.output_dim, random_state=0, batch_size=100, max_iter=100, verbose=False)
        scale = StandardScaler()

        # we define the feature extractor providing the model
        extractor = FeatureExtractor(feature_extractor=feature_extractor,
                                     model=model,
                                     scale=scale,
                                     out_dim=args.output_dim)

        # we fit the KMeans clustering model
        extractor.fit_model(training_paths)
        
        extractor.fit_scaler(training_paths)
        # now we can use features
        # we get query features
        query_features = extractor.extract_features(query_paths)
        query_features = extractor.scale_features(query_features)

        # we get gallery features
        gallery_features = extractor.extract_features(gallery_paths)
        gallery_features = extractor.scale_features(gallery_features)

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
