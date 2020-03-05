import csv
import os
import argparse
import glob
import nrrd
import re
import numpy as np
import pandas
from sklearn import preprocessing as sklearn_preprocessing

ACCEPTED_FILENAMES = [
    "2.seg.nrrd",
    "2.nrrd",
    "1.nrrd",
    "imagingVolume.nrrd",
    "segMask_tumor.nrrd",
    "segMask_tumor.seg.nrrd",
]

def calculate_length(vector):
    out = 0
    for v in vector:
        out += float(v) ** 2
    return np.sqrt(out)

def calculate_voxel_size(vectors):
    out = 1
    for vector in vectors:
        if vector == "none":
            continue
        out *= calculate_length(vector)
    return out

def calculate_volume(filename):
    try:
        array, metadata = nrrd.read(filename)
        unit_volume = array.sum()
        space_directions = metadata.get('space directions')
        if space_directions is None:
            print("{} has no space directions!".format(filename))
            return
        voxel_size = calculate_voxel_size(space_directions)
        return unit_volume * voxel_size
    except Exception as e:
        print(filename, e)

clinical_feature_functions = {
    "outcome": lambda f: f["outcome_2c.mod"],
    "outcome3": lambda f: f["outcome_3c.mod"],
    "premenorrhagia": lambda f: f["menorrhagia"] == "yes",
    "prebulk": lambda f: f["bulk"] == "yes",
    "preg": lambda f: int(f["preg_hist"]),
    "surg": lambda f: int(f["surgical_history"]),
    "followup": lambda f: float(f["mri_interval"]) if f["mri_interval"] != "" else None,
    "followup_raw": lambda f: float(f["mri_interval"]) if f["mri_interval"] != "" else None,
    "post-embolization-symptoms-binary": lambda f: f["postembolizationsymptoms_2c"] == "yes",
}

def clinical_features(feat, filename):
    patient = filename_features(filename)["patient"]
    clinical = feat.get(patient, None)
    if clinical is None:
        print("missing from clinical feature sheet: {}".format(patient))
        return {}
    return { k: f(clinical) for k, f in clinical_feature_functions.items() }

def statistics(df, shrink_cutoff=-0.10):
    df = df.assign(absolute_delta = df["volume"]["POST"] - df["volume"]["PRE"])
    df = df.assign(relative_change = (df["volume"]["POST"] - df["volume"]["PRE"])/df["volume"]["PRE"])
    df = df.assign(ratio = df["volume"]["POST"]/df["volume"]["PRE"])
    df = df.assign(pre_volume = df["volume"]["PRE"])
    df = df.assign(shrunk = df["relative_change"] < shrink_cutoff)
    return df

def all_nrrd(folder="."):
    return glob.glob("{}/**/*.nrrd".format(folder), recursive=True)

def all_features(filename="./features.csv"):
    with open(filename) as f:
        l = [ {k.lower(): v.lower() for k, v in row.items() } for row in csv.DictReader(f, skipinitialspace=True )]
        by_accession = { d["mrn"]: d for d in l }
        return by_accession

def filename_features(path):
    split_path = path.split(os.sep)
    filename = split_path[-1]
    modality = split_path[-2]
    pre_post = split_path[-3]
    if pre_post.lower() != "post":
        pre_post = "PRE"
    else:
        pre_post = "POST"
    accession = split_path[-4]
    patient = accession.split("-")[0]
    return {
        "accession": accession,
        "patient": patient,
        "pre_post": pre_post,
        "modality": modality,
        "filename": filename,
        "path": path,
    }

def filter_filenames(df):
    df = df[df.filename.isin(ACCEPTED_FILENAMES)]
    return df

def preprocessing(df):
    df = df[df.pre_post == "PRE"].drop(columns=["pre_post", "volume"])
    df = df.set_index(["accession",  "filename", "modality",]).unstack().unstack()
    return df

def preprocessing_post_files(df):
    df = df[df.pre_post == "POST"].drop(columns=["pre_post", "volume"])
    df = df.set_index(["accession",  "filename", "modality",]).unstack().unstack()
    return df

def features(df):
    f = df.set_index(["accession", "modality", "filename", "pre_post"])[["volume"]]
    f = f.unstack()
    f = statistics(f)
    f = f.reset_index()
    f = f.dropna()
    f = f[f.modality=="T1C"][f.filename=="segMask_tumor.nrrd"]
    f = f.set_index("accession")
    # for determining shrunk volume, use T1C only
    df = df[df.pre_post == "PRE"].drop(columns=["pre_post", "volume"])
    df = df[df.modality=="T1C"][df.filename=="segMask_tumor.nrrd"][["accession", "patient", *list(clinical_feature_functions.keys())]]
    df = df.set_index("accession")
    df = pandas.merge(df, pandas.DataFrame(f["shrunk"]), left_index=True, right_index=True)
    df = pandas.merge(df, pandas.DataFrame(f["pre_volume"]), left_index=True, right_index=True)
    df = pandas.merge(df, pandas.DataFrame(f["absolute_delta"]), left_index=True, right_index=True)
    df = pandas.merge(df, pandas.DataFrame(f["relative_change"]), left_index=True, right_index=True)
    df = df.dropna()
    return df

def normalize_column(df, column=""):
    min_max_scaler = sklearn_preprocessing.MinMaxScaler()
    x = df[[column]].values.astype(float)
    x_scaled = min_max_scaler.fit_transform(x)
    x_scaled = list(zip(*list(x_scaled)))[0]
    df[column] = pandas.Series(x_scaled, index=df.index)
    return df

def run(folder, features_filename, out):
    nrrds = all_nrrd(folder)
    feat = all_features(features_filename)
    # create all features
    nrrd_features = pandas.DataFrame(
        [{
            **filename_features(n),
            **clinical_features(feat, n),
            "volume": calculate_volume(n),
        } for n in nrrds])
    nrrd_features = filter_filenames(nrrd_features)
    nrrd_features = nrrd_features.dropna()
    nrrd_features = normalize_column(nrrd_features, column="followup")

    nrrd_features.to_csv(os.path.join(out, "nrrd-features.csv"), header=False)
    nrrd_features.to_pickle(os.path.join(out, "nrrd-features.pkl"))

    features_to_use = features(nrrd_features)
    features_to_use = normalize_column(features_to_use, column="pre_volume")
    features_to_use.to_csv(os.path.join(out, "training-features.csv"), header=False)
    features_to_use.to_pickle(os.path.join(out, "training-features.pkl"))

    to_preprocess = preprocessing(nrrd_features)
    to_preprocess.to_csv(os.path.join(out, "preprocess.csv"), header=False)
    to_preprocess.to_pickle(os.path.join(out, "preprocess.pkl"))

    to_preprocess_post_files = preprocessing_post_files(nrrd_features)
    to_preprocess_post_files.to_csv(os.path.join(out, "preprocess-post.csv"), header=False)
    to_preprocess_post_files.to_pickle(os.path.join(out, "preprocess-post.pkl"))

    return nrrd_features, features_to_use, to_preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='raw folder directory')
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='csv file of features')
    parser.add_argument(
        '--out',
        type=str,
        default="features",
        help='output folder')
    FLAGS, unparsed = parser.parse_known_args()
    run(FLAGS.folder, FLAGS.features, FLAGS.out)
