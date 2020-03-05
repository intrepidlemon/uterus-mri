import uuid
import os
import numpy as np
import pandas
import nrrd
import glob
import argparse
import random
from PIL import Image
import csv
from shutil import rmtree
from collections import defaultdict
from keras.preprocessing.image import ImageDataGenerator, Iterator
from tqdm import tqdm

from segmentation import calculate_percentile_slice, select_slice, bounding_box, crop, resize
from config import config

from filenames import IMAGE, SEGMENTATION, T1, T2

clinical_features = [
    "premenorrhagia",
    "prebulk",
    "preg",
    "surg",
    "followup",
    "pre_volume",
]

def all_input(t1, t2, features, labels):
    t1_image = np.array(t1)
    t1_image = np.rollaxis(t1_image, 0, 3)
    t2_image = np.array(t2)
    t2_image = np.rollaxis(t2_image, 0, 3)
    return (t1_image, t2_image), features, labels

def t1_input(t1, t2, features, labels):
    t1_image = np.array(t1)
    t1_image = np.rollaxis(t1_image, 0, 3)
    return (t1_image, None), [], labels

def t2_input(t1, t2, features, labels):
    t2_image = np.array(t2)
    t2_image = np.rollaxis(t2_image, 0, 3)
    return (None, t2_image), [], labels

def t1_t2_input(t1, t2, features, labels):
    t1_image = np.array(t1)
    t1_image = np.rollaxis(t1_image, 0, 3)
    t2_image = np.array(t2)
    t2_image = np.rollaxis(t2_image, 0, 3)
    return (t1_image, t2_image), [], labels

def t1_features_input(t1, t2, features, labels):
    t1_image = np.array(t2)
    t1_image = np.rollaxis(t1_image, 0, 3)
    return (t1_image, None), features, labels

def t2_features_input(t1, t2, features, labels):
    t2_image = np.array(t2)
    t2_image = np.rollaxis(t2_image, 0, 3)
    return (t2_image, None), features, labels

def features_input(t1, t2, features, labels):
    return (None, None), features, labels

INPUT_FORMS = {
    "all": all_input,
    "t1": t1_input,
    "t2": t2_input,
    "t1-t2": t1_t2_input,
    "t1-features": t1_features_input,
    "t2-features": t2_features_input,
    "features": features_input,
}

INPUT_FORM_PARAMETERS = {
    "all": {
        "t1": True,
        "t2": True,
        "features": True,
    },
    "t1": {
        "t1": True,
        "t2": False,
        "features": False,
    },
    "t2": {
        "t1": False,
        "t2": True,
        "features": False,
    },
    "t1-t2": {
        "t1": True,
        "t2": True,
        "features": False,
    },
    "t1-features": {
        "t1": True,
        "t2": False,
        "features": True,
    },
    "t2-features": {
        "t1": False,
        "t2": True,
        "features": True,
    },
    "features": {
        "t1": False,
        "t2": False,
        "features": True,
    },
}

class Features(Iterator):
    def __init__(self, features, shuffle, seed, batch_size=config.BATCH_SIZE):
        super(Features, self).__init__(len(features), batch_size, shuffle, hash(seed) % 2**32 )
        self.features = np.array(features)

    def _get_batches_of_transformed_samples(self, index_array):
        return self.features[index_array]

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

class BalancedDataset(object):
    def __init__(self, images, features, labels, names, augment=False, shuffle=False, seed=None, input_form="all", batch_size=config.BATCH_SIZE, ratios=None):

        self.seed = seed
        self.augment = augment
        self.input_form = input_form
        self.names = names
        self.batch_size = batch_size

        self.parameters = INPUT_FORM_PARAMETERS[input_form]

        features = list(zip(*features))

        self.labels = labels

        self.features = features
        self.n = len(labels)

        unique, index, inverse, counts = np.unique(self.labels, return_index=True, return_inverse=True, return_counts=True)
        self.y = inverse
        self.classes = inverse
        self.class_indices = { u: i for i, u in enumerate(unique) }

        self.features_size = 0
        if self.parameters["features"]:
            self.features_size = len(features[0])

        self.batch_sizes = { c: int(batch_size/len(unique)) for c in unique }
        if ratios is not None:
            if type(ratios) != dict or sum(ratios.values()) != 1 or not all([(c in ratios) for c in unique ]):
                print("error ratios are off")
            else:
                self.batch_sizes = { batch_size * ratios.get(c) for c in unique }

        self.datasets = dict()
        for c in unique:
            current_images = [image for i, image in enumerate(images) if labels[i] == c]
            current_features = list(zip(*[feature for i, feature in enumerate(features) if labels[i] == c]))
            current_labels = [label for i, label in enumerate(labels) if labels[i] == c]
            current_names = [name for i, name in enumerate(names) if labels[i] == c]

            self.datasets[c] = Dataset(
                current_images,
                current_features,
                current_labels,
                current_names,
                augment=augment,
                shuffle=shuffle,
                seed=seed,
                input_form=input_form,
                batch_size=self.batch_sizes[c],
                class_start_index=self.class_indices[c],
            )

    def __len__(self):
        return self.n

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        for dataset in self.datasets.values():
            dataset.reset()

    def next(self):
        inputs, labels = zip(*([d.next() for d in self.datasets.values()]))
        labels = np.concatenate(labels)
        if type(inputs[0]) == list:
            inputs = zip(*inputs)
            inputs = [np.concatenate(i) for i in inputs]
            combined = list(zip(*inputs, labels))
        else:
            inputs = list(zip(*inputs))
            inputs = np.concatenate(inputs)
            combined = list(zip(inputs, labels))
        random.shuffle(combined)
        inputs = list(zip(*combined))
        labels = np.array(inputs.pop())
        inputs = [np.array(i) for i in inputs]
        if len(inputs) == 1:
            inputs = inputs[0]
        return inputs, labels

class Dataset(object):
    def __init__(self, images, features, labels, names, augment=False, shuffle=False, seed=None, input_form="all", batch_size=config.BATCH_SIZE, class_start_index=0):
        self.shuffle = shuffle
        self.seed = seed
        self.augment = augment
        self.input_form = input_form
        self.names = names
        self.batch_size = batch_size

        self.parameters = INPUT_FORM_PARAMETERS[input_form]

        features = list(zip(*features))

        self.labels = labels

        self.features = features
        self.features_size = 0
        if self.parameters["features"]:
            self.features_size = len(features[0])
            self.features_generator = Features(self.features, self.shuffle, self.seed, self.batch_size)

        self.n = len(labels)

        unique, index, inverse, counts = np.unique(self.labels, return_index=True, return_inverse=True, return_counts=True)
        inverse = [i + class_start_index for i in inverse]
        self.y = inverse
        self.classes = inverse
        self.class_indices = { u: i + class_start_index for i, u in enumerate(unique) }

        separate_images = list(zip(*images))
        if self.parameters["t1"]:
            self.t1 = np.array(separate_images[0])
            self.datagen = self._get_data_generator()

        if self.parameters["t2"]:
            self.t2 = np.array(separate_images[1])
            self.datagen2 = self._get_data_generator()

        self.reset()

    def __len__(self):
        return self.n

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        if self.parameters["features"]:
            self.features_generator = Features(self.features, self.shuffle, self.seed, self.batch_size)

        if self.parameters["t1"]:
            self.generator_t1 = self.datagen.flow(
                    x=self.t1,
                    y=self.y,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    seed=hash(self.seed) % 2**32,
                    )

        if self.parameters["t2"]:
            self.generator_t2 = self.datagen2.flow(
                    x=self.t2,
                    y=self.y,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    seed=hash(self.seed) % 2**32 ,
                    )
        self.labels_generator = Features(self.y, self.shuffle, self.seed, self.batch_size)

    def _get_data_generator(self):
        if self.augment:
            return ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
            )
        return ImageDataGenerator(
            rescale=1. / 255,
            )

    def next(self):
        labels = self.labels_generator.next()
        inputs = list()
        if self.parameters["t2"]:
            inputs.append(self.generator_t2.next()[0])
        if self.parameters["t1"]:
            inputs.append(self.generator_t1.next()[0])
        if self.parameters["features"]:
            inputs.append(self.features_generator.next())
        if len(inputs) == 1:
            inputs = inputs[0]
        return (inputs, labels)

def shrunk_feature(row):
    label = "shrunk" if row["shrunk"] else "not-shrunk"
    features = [ row[f] for f in clinical_features ]
    return label, features

def outcome_feature(row):
    label = row["outcome"]
    features = [ row[f] for f in clinical_features ]
    return label, features

LABEL_FORMS = {
    "shrunk": shrunk_feature,
    "outcome": outcome_feature,
}

def get_label_features(row, label="shrunk"):
    """returns label, features, sample name"""
    return (*LABEL_FORMS[label](row), row.name)

def input_data_form(t1, t2, features, labels, input_form=config.INPUT_FORM):
    images, features, labels = INPUT_FORMS[input_form](t1, t2, features, labels)
    return images, features, labels

def load_image(image_path, segmentation_path, verbose=False):
    image, _ = nrrd.read(image_path)
    segmentation, _ = nrrd.read(segmentation_path)
    if verbose:
        print("""
        image: {}
        seg: {}
""".format(image.shape, segmentation.shape))
    return [mask_image_percentile(image, segmentation, 100, a) for a in (0, 1, 2)]

def mask_image_percentile(image, segmentation, percentile=100, axis=2):
    plane = calculate_percentile_slice(segmentation, percentile, axis)
    image, segmentation = select_slice(image, segmentation, plane, axis)

    bounds = bounding_box(segmentation)
    image, segmentation = crop(image, segmentation, bounds)

    masked = image * segmentation
    masked = resize(masked, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    return masked

SHAPES_OUTPUT = """
SHAPES
    {}:"""

def generate_from_features(df, input_form=config.INPUT_FORM, label_form="shrunk", verbose=False):
    source = config.PREPROCESSED_DIR
    parameters = INPUT_FORM_PARAMETERS[input_form]

    for index, row in tqdm(df.iterrows(), total=len(df)):
        t1_image_file = os.path.join(source, "{}-{}-{}".format(index, T1, IMAGE))
        t1_seg_file = os.path.join(source, "{}-{}-{}".format(index, T1, SEGMENTATION))
        t2_image_file = os.path.join(source, "{}-{}-{}".format(index, T2, IMAGE))
        t2_seg_file = os.path.join(source, "{}-{}-{}".format(index, T2, SEGMENTATION))
        try:
            t1_masked = None
            t2_masked = None
            if parameters["t1"] or parameters["features"]: # load in case of features so that files that error out aren't included in analysis
                if verbose:
                    print(SHAPES_OUTPUT.format("t1"))
                t1_masked = load_image(t1_image_file, t1_seg_file, verbose=verbose)
            if parameters["t2"]:
                if verbose:
                    print(SHAPES_OUTPUT.format("t2"))
                t2_masked = load_image(t2_image_file, t2_seg_file, verbose=verbose)
            labels, features, name = get_label_features(row, label=label_form)
            images, features, labels = input_data_form(t1_masked, t2_masked, features, labels, input_form=input_form)
            yield images, features, labels, name

        except Exception as e:
            print()
            print("#" * 80)
            print("Exception occured for: {}\n{}".format(row, e))
            continue

def sort(validation_fraction=0.2, test_fraction=0.1, seed=None, label_form="shrunk"):
    f = pandas.read_pickle(config.FEATURES)

    all_train = list()
    all_validation = list()
    all_test = list()

    for label in f[label_form].unique():
        label_set = f[f[label_form] == label]

        validation_label_set = label_set.sample(frac=validation_fraction, random_state=(int(seed) % 2 ** 32))
        label_set = label_set.drop(validation_label_set.index)
        test_label_set = label_set.sample(frac=(test_fraction/(1-validation_fraction)), random_state=(int(seed) % 2 ** 32))
        label_set = label_set.drop(test_label_set.index)

        all_train.append(label_set)
        all_validation.append(validation_label_set)
        all_test.append(test_label_set)

    train = pandas.concat(all_train)
    validation = pandas.concat(all_validation)
    test = pandas.concat(all_test)

    train.to_csv(os.path.join(config.DATASET_RECORDS, "{}-train.csv".format(str(seed))))
    validation.to_csv(os.path.join(config.DATASET_RECORDS, "{}-validation.csv".format(str(seed))))
    test.to_csv(os.path.join(config.DATASET_RECORDS, "{}-test.csv".format(str(seed))))

    return train, validation, test

def relist(l):
    l = list(l)
    if len(l) == 0:
        return l
    return [[k[i] for k in l] for i, _ in enumerate(l[0])]

def data(seed=None,
        input_form=config.INPUT_FORM,
        label_form="shrunk",
        train_shuffle=True,
        validation_shuffle=False,
        test_shuffle=False,
        train_augment=True,
        validation_augment=False,
        test_augment=False,
        validation_split=config.VALIDATION_SPLIT,
        test_split=config.TEST_SPLIT,
        verbose=False,
        balanced_train=False,
        ):
    train, validation, test = sort(validation_split, test_split, seed, label_form)
    train_images, train_features, train_labels, train_names = relist(generate_from_features(train, input_form=input_form, label_form=label_form, verbose=verbose))
    validation_images, validation_features, validation_labels, validation_names = relist(generate_from_features(validation, input_form=input_form, label_form=label_form, verbose=verbose))
    test_images, test_features, test_labels, test_names = relist(generate_from_features(test, input_form=input_form, label_form=label_form, verbose=verbose))

    train_features = relist(train_features)
    validation_features = relist(validation_features)
    test_features = relist(test_features)


    if balanced_train:
        train_generator = BalancedDataset(
                train_images,
                train_features,
                train_labels,
                train_names,
                augment=train_augment,
                shuffle=train_shuffle,
                input_form=input_form,
                seed=seed,
            )
    else:
        train_generator = Dataset(
                train_images,
                train_features,
                train_labels,
                train_names,
                augment=train_augment,
                shuffle=train_shuffle,
                input_form=input_form,
                seed=seed,
            )
    validation_generator = Dataset(
            validation_images,
            validation_features,
            validation_labels,
            validation_names,
            augment=validation_augment,
            shuffle=validation_shuffle,
            input_form=input_form,
            seed=seed,
        )
    test_generator = Dataset(
            test_images,
            test_features,
            test_labels,
            test_names,
            augment=test_augment,
            shuffle=test_shuffle,
            input_form=input_form,
            seed=seed,
        )
    return train_generator, validation_generator, test_generator

def xdata(fold_number,
          train,
          validation,
          test,
          # holdout_test,
          seed=None,
          input_form=config.INPUT_FORM,
          label_form="shrunk",
          train_shuffle=True,
          validation_shuffle=False,
          test_shuffle=False,
          train_augment=True,
          validation_augment=False,
          test_augment=False,
          verbose=False
          ):

    #save the data in each set for the fold run
    fold_string = 'fold-' + str(fold_number)
    train.to_csv(os.path.join(config.DATASET_RECORDS, fold_string + "-{}-ktrain.csv".format(str(seed))))
    validation.to_csv(os.path.join(config.DATASET_RECORDS, fold_string + "-{}-kvalidation.csv".format(str(seed))))
    test.to_csv(os.path.join(config.DATASET_RECORDS, fold_string + "-{}-ktest.csv".format(str(seed))))
    #holdout_test.to_csv(os.path.join(config.DATASET_RECORDS, fold_string + "-{}-kholdouttest.csv".format(str(seed))))

    # loading of the features - this is supposed to be the bottleneck, but seems to be pretty fast when I was testing it; refactor later
    train_images, train_features, train_labels, train_names = relist(generate_from_features(train, input_form=input_form, label_form=label_form, verbose=verbose))
    validation_images, validation_features, validation_labels, validation_names = relist(generate_from_features(validation, input_form=input_form, label_form=label_form, verbose=verbose))
    test_images, test_features, test_labels, test_names = relist(generate_from_features(test, input_form=input_form, label_form=label_form, verbose=verbose))
    # holdouttest_images, holdouttest_features, holdouttest_labels, holdouttest_names = relist(generate_from_features(holdout_test, input_form=input_form, label_form=label_form, verbose=verbose))

    train_features = relist(train_features)
    validation_features = relist(validation_features)
    test_features = relist(test_features)
    # holdouttest_features = relist(holdouttest_features)

    train_generator = Dataset(
            train_images,
            train_features,
            train_labels,
            train_names,
            augment=train_augment,
            shuffle=train_shuffle,
            input_form=input_form,
            seed=seed,
        )
    validation_generator = Dataset(
            validation_images,
            validation_features,
            validation_labels,
            validation_names,
            augment=validation_augment,
            shuffle=validation_shuffle,
            input_form=input_form,
            seed=seed,
        )
    test_generator = Dataset(
            test_images,
            test_features,
            test_labels,
            test_names,
            augment=test_augment,
            shuffle=test_shuffle,
            input_form=input_form,
            seed=seed,
        )
    # holdout_test_generator = Dataset(holdouttest_images,holdouttest_features,holdouttest_labels,holdouttest_names,augment=test_augment,shuffle=test_shuffle,input_form=input_form,seed=seed,)
    return train_generator, validation_generator, test_generator  # , holdout_test_generator

if __name__ == '__main__':
    data(uuid.uuid4())
