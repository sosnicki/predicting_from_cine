import pathlib
import pickle
import re
import traceback
from collections import defaultdict

import cv2
import joblib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pymongo
from django.conf import settings
from django.core.management import BaseCommand
from matplotlib.backends.backend_pdf import PdfPages
from pineai.db import collection_by_name, PreprocessingCollection, collection_by_meta, Document
from skimage.morphology import skeletonize
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from serce.transformers.prepare import PrepareTransformer
from serce.transformers.radiomics_features import RadiomicsFeatureTransformer


def process_img(data, model_path, source):
    try:
        clf = pickle.loads(model_path.read_bytes())
        radiomics_transformer = RadiomicsFeatureTransformer()
        prepare_transformer = PrepareTransformer(source)

        doc = Document(data=data)
        radiomics_transformer.transform_doc(doc)
        prepare_transformer.transform_doc(doc)

        X = np.array([doc['X']])
        # print(f'Predicting X {X.shape}')
        y_true_prob = clf.predict_proba(X)
        y_true = clf.predict(X)

        print(y_true_prob[0], y_true[0])
        return y_true_prob[0], y_true[0]
    except:
        traceback.print_exc()


class Command(BaseCommand):
    model_path: pathlib.Path
    mask_size: int
    source: str
    method: str

    def add_arguments(self, parser):
        parser.add_argument('action')
        parser.add_argument('--mask_size', default=30, type=int)
        parser.add_argument('--source')
        parser.add_argument('--method')

    def handle(self, *args, **options):
        self.mask_size = options['mask_size']
        self.source = options['source']
        self.method = options['method']
        self.model_path = settings.SOURCE_DATA_DIR / f'model_{self.source}_{self.method}_{self.mask_size}.pkl'

        action = options['action']
        if action == 'load':
            self.load()
        elif action == 'draw_roc':
            self.draw_roc()
        elif action == 'draw':
            self.draw()
        elif action == 'load_npy':
            self.load_npy()
        elif action == 'build':
            self.build()

    def load_npy(self):
        data = []
        y_true, y_pred, y_prob = [], [], []
        for src in ['artifacts_test', 'negative_no_LGE', 'negative_test', 'positive_test']:
            samples = np.load(settings.SOURCE_DATA_DIR / f'{src}.npy', allow_pickle=True)
            results = joblib.Parallel(backend='multiprocessing', n_jobs=-1)(
                joblib.delayed(process_img)(d, self.model_path, self.source) for d in samples)

            for nr, (r, d) in enumerate(zip(results, samples)):
                row = {
                    'nr': nr,
                    'true label': d['label'],
                    'src': src,
                }
                if r is not None:
                    prob, label = r
                    row.update({
                        'pred label': label,
                        'prob 0': prob[0],
                        'prob 1': prob[1],
                    })
                    y_true.append(d['label'])
                    y_pred.append(label)
                    y_prob.append(prob[1])
                data.append(row)
        df = pd.DataFrame(data)
        df.to_csv(settings.ANALYSIS_DIR / 'test_npy.csv')

        print(f'Accuracy: {accuracy_score(y_true, y_pred):0.6f}')
        print(f'ROC AUC prob: {roc_auc_score(y_true, y_prob):0.6f}')
        print(f'ROC AUC label: {roc_auc_score(y_true, y_pred):0.6f}')

    def load_nii_dir(self, dir_path):
        rx = re.compile(r'image_(\d+)[_\.]')
        images = {}
        for img_path in dir_path.iterdir():
            img_nr = int(rx.search(img_path.name).group(1))
            images[img_nr] = img_path
        return images

    def calc_slice(self, half_box, w, h, cine, mask, opt_flow, reg_tran):
        skeleton = skeletonize(np.copy(mask)).astype(np.uint8)
        prob = np.zeros((h, w))
        mask_sum = np.zeros((h, w))
        patches = []

        # pomijanie pustych masek
        if np.sum(skeleton) == 0:
            return skeleton, prob, patches, mask_sum
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv2.contourArea)

        for point in largest_contour:
            cy = point[0][1]
            cx = point[0][0]
            roi = np.s_[cy - half_box:cy + half_box, cx - half_box:cx + half_box]
            patch = {
                'cx': int(cx),
                'cy': int(cy),
            }
            if cx - half_box >= 0 and cx + half_box < w and cy - half_box >= 0 and cy + half_box < h:
                patch.update({
                    'cine': cine[roi],
                    'mask': mask[roi],
                    'optical_flow': opt_flow[roi],
                    'registration_transform': reg_tran[roi],
                })
            patches.append(patch)
        results = joblib.Parallel(backend='multiprocessing', n_jobs=-1)(
            joblib.delayed(process_img)(d, self.model_path, self.source) for d in patches)
        # results = [process_img(d, self.model_path) for d in docs]

        for r, doc in zip(results, patches):
            if r is None:
                continue
            y_prob, label = r
            cx = doc['cx']
            cy = doc['cy']
            roi = np.s_[cy - half_box:cy + half_box, cx - half_box:cx + half_box]
            prob[roi] += y_prob[1] * doc['mask']
            mask_sum[roi] += doc['mask']
        prob[mask_sum > 0] /= mask_sum[mask_sum > 0]
        return skeleton, prob, patches, mask_sum

    def flatten_dist(self, img):
        return np.sqrt(np.square(img[:, :, 0]) + np.square(img[:, :, 1]))

    def draw_slices(self, nr, docs):
        pdf = PdfPages(str(settings.ANALYSIS_DIR / f'pdf/{self.source}_{self.method}_{nr}.pdf'))
        for doc in docs:
            fig, axs = plt.subplots(1, 6, figsize=(4000 / 100, 600 / 100), dpi=100)

            axs[0].imshow(doc["cine"], cmap='gray')
            axs[0].set_title(f'Cine {doc["slice_nr"]}')
            axs[1].imshow(doc["mask"], cmap='gray')
            axs[1].set_title(f'Mask {doc["slice_nr"]}')
            axs[2].imshow(self.flatten_dist(doc["opt_flow"]), cmap='gray')
            axs[2].set_title(f'Optical flow {doc["slice_nr"]}')
            axs[3].imshow(self.flatten_dist(doc["reg_tran"]), cmap='gray')
            axs[3].set_title(f'Registration transform {doc["slice_nr"]}')
            axs[4].imshow(doc["skeleton"], cmap='gray')
            axs[4].set_title(f'Skeleton {doc["slice_nr"]}')
            axs[5].imshow(doc["prob"], cmap='gray')
            axs[5].set_title(f'Probability {doc["slice_nr"]}')
            fig.tight_layout()
            pdf.savefig()
        pdf.close()

    def draw_patches(self, nr, docs):
        pdf = PdfPages(str(settings.ANALYSIS_DIR / f'pdf/{self.source}_{self.method}_{nr}_patches.pdf'))
        for doc in docs:
            for patch in doc['patches']:
                fig, axs = plt.subplots(1, 4, figsize=(2000 / 100, 500 / 100), dpi=100)
                axs[0].imshow(patch['cine'], cmap='gray')
                axs[0].set_title(f'Cine {doc["slice_nr"]}')
                axs[1].imshow(patch['mask'], cmap='gray')
                axs[1].set_title(f'Mask {doc["slice_nr"]}')
                axs[2].imshow(self.flatten_dist(patch['optical_flow']), cmap='gray')
                axs[2].set_title(f'Optical flow {doc["slice_nr"]}')
                axs[3].imshow(self.flatten_dist(patch['registration_transform']), cmap='gray')
                axs[3].set_title(f'Registration transform {doc["slice_nr"]}')
                fig.tight_layout()
                pdf.savefig()
                plt.close('all')
        pdf.close()

    def draw(self):
        docs_map = defaultdict(list)
        for doc in collection_by_name('classif').find_docs({
            'source': self.source,
            'mask_size': self.mask_size,
            'method': self.method,
        }):
            docs_map[doc['nr']].append(doc)
        for nr, docs in docs_map.items():
            self.draw_slices(nr, docs)
            self.draw_patches(nr, docs)

    def draw_roc(self):
        max_means = []
        means = []
        labels = []
        for lge_label in [0, 1]:
            docs_map = defaultdict(list)
            for doc in collection_by_name('classif').find_docs({
                'source': self.source,
                'mask_size': self.mask_size,
                'method': self.method,
                'lge_label': lge_label
            }):
                docs_map[doc['nr']].append(doc)
            for nr, docs in docs_map.items():
                slices_mean = np.array([np.mean(d['prob'][d['mask_sum'] > 0]) for d in docs])
                slices_mean = slices_mean[~np.isnan(slices_mean)]
                max_mean = np.max(slices_mean)
                mean = np.mean(slices_mean)

                max_means.append(max_mean)
                means.append(mean)
                labels.append(lge_label)

        print(f'Labels: {labels}')
        print(f'Max means: {max_means}')
        print(f'Means: {means}')
        score = roc_auc_score(labels, max_means)
        print(f'Max means score: {score}')
        score = roc_auc_score(labels, means)
        print(f'Means score: {score}')

    def load(self):
        half_box = self.mask_size // 2

        clf = pickle.loads(self.model_path.read_bytes())
        print(f'Classes: {clf.classes_}')
        coll = collection_by_name('classif')
        coll.clear()

        for lge_dir, lge_label in [('LGE', 1), ('No_LGE', 0)]:
            cines = self.load_nii_dir(settings.SOURCE_DATA_DIR / f'CINE_Ts/{lge_dir}/imagesTs')
            seg_cines = self.load_nii_dir(settings.SOURCE_DATA_DIR / f'CINE_Ts/{lge_dir}/predictionsTs')
            reg_transforms = self.load_nii_dir(
                settings.SOURCE_DATA_DIR / f'CINE_Ts/{lge_dir}/transformationTs_heartContracted')
            optical_flows = self.load_nii_dir(
                settings.SOURCE_DATA_DIR / f'CINE_Ts/{lge_dir}/opticalFlowTs_heartContracted')

            ids = set(cines) & set(seg_cines) & set(reg_transforms) & set(optical_flows)

            for nr in sorted(ids):
                cine_slices = nib.load(cines[nr]).get_fdata()
                seg_slices = nib.load(seg_cines[nr]).get_fdata()
                opt_flow_slices = nib.load(optical_flows[nr]).get_fdata()
                reg_tran_slices = nib.load(reg_transforms[nr]).get_fdata()

                mask_slices = np.copy(seg_slices)
                mask_slices[mask_slices == 2] = 0
                mask_slices[mask_slices != 0] = 1
                mask_slices = np.ascontiguousarray(mask_slices.astype(np.uint8))

                h, w, slices_count = cine_slices.shape

                for slice_nr in range(slices_count):
                    cine = cine_slices[:, :, slice_nr]
                    mask = mask_slices[:, :, slice_nr]
                    opt_flow = opt_flow_slices[:, :, slice_nr, :]
                    reg_tran = reg_tran_slices[:, :, slice_nr, :]
                    skeleton, prob, patches, mask_sum = self.calc_slice(half_box, w, h, cine, mask, opt_flow, reg_tran)
                    coll.new_doc({
                        'name': f'{nr}_{slice_nr}_{self.source}_{self.method}_{self.mask_size}',
                        'source': self.source,
                        'mask_size': self.mask_size,
                        'method': self.method,
                        'nr': nr,
                        'slice_nr': slice_nr,
                        'cine': cine,
                        'mask': mask,
                        'opt_flow': opt_flow,
                        'reg_tran': reg_tran,
                        'skeleton': skeleton,
                        'prob': prob,
                        'patches': np.array(patches),
                        'mask_sum': mask_sum,
                        'lge_label': lge_label
                    }).save()

    def build(self):
        radiomics_coll = PreprocessingCollection.filter(transformer='radiomicsfeature',
                                                        parent=f'{self.mask_size}x{self.mask_size}')[0]
        meta_info = collection_by_name('meta').find_one({
            "parent": radiomics_coll.meta_name,
            "pipeline.params.source": self.source,
            "pipeline.params.kbest_k": None,
            "pipeline.params.pca": None
        })
        coll = collection_by_meta(meta_info)
        vect_ids = [d['_id'] for d in collection_by_name('vect').find({'cin': coll.meta_name})]
        opt_docs = collection_by_name('opt').find_docs({'din.$id': {'$in': vect_ids},
                                                        'methods': [self.method]},
                                                       sort=[('roc_auc', pymongo.DESCENDING)], limit=10)

        params = []
        for opt_doc in opt_docs:
            # print(opt_doc['roc_auc'])
            # print(opt_doc['best_params'])
            params.append(opt_doc['best_params'])
        params_df = pd.DataFrame(params)
        # print(params_df)
        best_params = {}
        for k in params_df.keys():
            try:
                v = params_df[k].mean().astype(params_df[k].dtype)
            except TypeError:
                v = params_df[k].value_counts(dropna=False).idxmax()
            best_params[k.split('__')[1]] = v

        print(f'Preparing dataset from collection {coll.pretty_name}')
        X, y = [], []
        for doc in coll.find_docs():
            X.append(doc['X'])
            y.append(doc['label'])

        print(f'Fitting {self.method} with params {best_params}')
        if self.method == 'GBC':
            clf = GradientBoostingClassifier(**best_params)
        elif self.method == 'RFB':
            clf = RandomForestClassifier(class_weight='balanced', **best_params)
        elif self.method == 'KNN':
            clf = KNeighborsClassifier(**best_params)
        else:
            raise Exception('Invalid method')
        X = np.array(X)
        y = np.array(y)
        clf.fit(X, y)
        print(f'Fitting model with X {X.shape} and y {y.shape}')
        self.model_path.write_bytes(pickle.dumps(clf))
        print(f'Model saved to path {self.model_path}')

        clf2 = pickle.loads(self.model_path.read_bytes())
        y_pred = clf2.predict(X)
        print(f'Accuracy: {accuracy_score(y, y_pred) * 100:0.2f}')
        print(f'ROC AUC: {roc_auc_score(y, y_pred) * 100:0.2f}')
