#!/usr/bin/env python
# coding: utf-8

# Headless-script compatibility shim: `display()` is a Jupyter builtin, not a
# real Python function, and the original notebook has no explicit import for
# it (nbconvert does not synthesize one). Running this script outside a
# kernel would otherwise crash the very first cell that calls display()
# (unguarded Step 2 cold-start block) with NameError. Fall back to printing
# the repr, which is all a plain terminal/log can show anyway.
try:
    from IPython.display import display  # noqa: F401
except Exception:
    def display(*objects):
        for obj in objects:
            print(obj)

# # StitchingNet Q1-Ready Full Experiment-v5 (Compact Deployment Validity)
#
# ⚠️  EDITED FOR LOCAL RESUME MODE (2026-09-24):
# - Step 1 (matched-budget LODO) COMMENTED OUT (already completed)
# - Step 2 (cold-start fixed query) RESUME MODE ENABLED (will skip completed runs)
# - Native progress monitoring added (real-time progress display)
# - Backup: stitchnet-v5-revision-r3-step2-fixed-query-coldstart-jupyterlab-full.py.backup
#
# This revision implements the Option A fixes from `V3_BRUTAL_AUDIT_20260710.md`:
# 
# - replaces the OOM-prone `Flatten -> Dense(1024)` head with `GlobalAveragePooling2D -> Dropout -> Dense(11)`
# - keeps MobileNetV1 as the deployment-validity backbone, but the compact head is used for all CNN backbones
# - writes LODO and cold-start rows incrementally without corrupting CSV schemas
# - fails publication readiness when any protocol table contains non-null `error` rows
# - requires metric columns for LODO and cold-start tables before declaring publication data ready
# - supports three deployment protocol seeds in full mode (`DEPLOYMENT_PROTOCOL_SEEDS = [0, 1, 2]`)
# - validates TFLite by running interpreter inference on a representative test subset and comparing against Keras
# - records Grad-CAM status explicitly; failed Grad-CAM can no longer be treated as explainability evidence
# 
# Kaggle settings: GPU on. T4x2 is helpful for availability, but the notebook is designed to remain memory-safe on a single active Keras process.
# 

# In[13]:


from pathlib import Path

RUN_MODE = 'full'
INSTALL_MLFLOW = False
ENABLE_MLFLOW = False
KAGGLE_COMPACT_ARTIFACTS = True
SAVE_INTERMEDIATE_KERAS_MODELS = False
SAVE_FINAL_KERAS_MODEL = True

DATASET_ROOT = Path('./kagglehub_cache/stitchingnet-dataset')  # placeholder; overwritten by kagglehub resolution in the next cell
PROJECT_DIR = Path('./stitchnet_q1_ready_v5_compact')
SPLIT_DIR = PROJECT_DIR / 'splits'
RESULTS_DIR = PROJECT_DIR / 'results'
MODEL_DIR = PROJECT_DIR / 'models'
FIGURE_DIR = PROJECT_DIR / 'figures'
MLFLOW_DIR = PROJECT_DIR / 'mlruns'
MLFLOW_DB = PROJECT_DIR / 'mlflow.db'
MLFLOW_ARTIFACT_DIR = PROJECT_DIR / 'mlflow_artifacts'
REPORT_DIR = PROJECT_DIR / 'reports'

for directory in [PROJECT_DIR, SPLIT_DIR, RESULTS_DIR, MODEL_DIR, FIGURE_DIR, MLFLOW_DIR, MLFLOW_ARTIFACT_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Bug fix (2026-09-27, post-mortem on the wiped comment-#7 LODO Grad-CAM
# checkpoints): this cleanup used to run unconditionally on every script
# execution and deleted every *.keras/.h5/.ckpt/.pb/.tflite under MODEL_DIR,
# including checkpoints intentionally saved by prior Round-3 R&R addendum
# runs (SAVE_MODEL_DOMAINS / SAVE_MODEL_DOMAINS_COLDSTART, comment #7). A
# resumed Step 1 run silently destroyed the Step 2 cold-start Grad-CAM
# checkpoints (and vice versa) because the cleanup cell re-fires at import
# time on every execution. Protected prefixes below are the addendum
# checkpoint-saving prefixes (`save_prefix='domain_matched'` / `'cold_fixedq'`)
# and must never be deleted by this routine.
PROTECTED_MODEL_PREFIXES = ('domain_matched_', 'cold_fixedq_')

if KAGGLE_COMPACT_ARTIFACTS:
    for pattern in ['*.keras', '*.h5', '*.ckpt', '*.pb', '*.tflite']:
        for path in MODEL_DIR.glob(pattern):
            if path.name.startswith(PROTECTED_MODEL_PREFIXES):
                print(f'KAGGLE_COMPACT_ARTIFACTS cleanup: preserving protected checkpoint {path.name}')
                continue
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                import shutil
                shutil.rmtree(path)
    for pattern in ['stitchnet_q1_ready_v5_compact_evidence_*.zip', 'stitchnet_q1_ready_compact_evidence_*.zip']:
        for path in PROJECT_DIR.parent.glob(pattern):
            if path.is_file():
                path.unlink()

PRIMARY_SEED = 0
SEEDS = [0, 1, 2] if RUN_MODE == 'full' else [0]
DEPLOYMENT_PROTOCOL_SEEDS = SEEDS
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
FOLD_BATCH_SIZE = 8
HEAD_EPOCHS = 30 if RUN_MODE == 'full' else 1
FINETUNE_EPOCHS = 40 if RUN_MODE == 'full' else 1
DOMAIN_EPOCHS = 8 if RUN_MODE == 'full' else 1
ADAPT_EPOCHS = 8 if RUN_MODE == 'full' else 1
HEAD_LR = 1e-3
FINETUNE_LR = 1e-5
DOMAIN_LR = 1e-5
ADAPT_LR = 1e-5

# Compact head: fixes the 50176 x 1024 dense allocation that caused V3/V4 LODO and cold-start OOM.
HEAD_DROPOUT = 0.30
HEAD_DENSE_UNITS = 0  # set >0 only if you intentionally want a compact bottleneck before classification

PHASE0_CNN_MODELS = ['DenseNet121', 'ResNet152V2', 'Xception', 'MobileNetV1', 'InceptionV3', 'VGG19']
DEPLOYMENT_VALIDITY_MODEL_NAME = 'MobileNetV1'
FINAL_SELECTION_POLICY = 'deployment_validity_backbone'  # alternatives: 'highest_f1'
DEPLOYMENT_MODEL_NAMES = [DEPLOYMENT_VALIDITY_MODEL_NAME]
SUPPORT_SIZES = [0, 1, 5, 10, 25, 50] if RUN_MODE == 'full' else [0, 1]
ADAPTATION_VAL_PER_CLASS = 5
ROBUSTNESS_PERTURBATIONS = ['clean', 'brightness_down', 'brightness_up', 'contrast_down', 'contrast_up', 'gaussian_blur', 'gaussian_noise', 'translate', 'low_resolution']
TFLITE_VALIDATION_MAX_IMAGES = 512 if RUN_MODE == 'full' else 64
GRADCAM_MAX_EXAMPLES = 6

print('RUN_MODE:', RUN_MODE)
print('PROJECT_DIR:', PROJECT_DIR)
print('DEPLOYMENT_VALIDITY_MODEL_NAME:', DEPLOYMENT_VALIDITY_MODEL_NAME)
print('DEPLOYMENT_PROTOCOL_SEEDS:', DEPLOYMENT_PROTOCOL_SEEDS)
print('HEAD: GlobalAveragePooling2D -> Dropout({}) -> Dense(num_classes)'.format(HEAD_DROPOUT))

# Round-3 R&R addendum: when False, skips the full backbone-sweep / original
# 33-run LODO / 198-run cold-start / robustness / calibration / edge / Grad-CAM /
# audit pipeline (cells already reproduced in the submitted evidence package) so
# a fresh Kaggle session only spends GPU quota on the new Step-1+ addendum cells
# below. Set True to fully reproduce the original pipeline from scratch instead.
RUN_FULL_ORIGINAL_PIPELINE = False
print('RUN_FULL_ORIGINAL_PIPELINE:', RUN_FULL_ORIGINAL_PIPELINE)


# Round-3 R&R addendum: Step 1 (matched-budget LODO) already completed
# successfully on Kaggle on 2026-09-17 (12/12 domain-seed runs, results in
# domain_generalisation_matched_budget_results.csv). Set to False so this
# JupyterLab kernel does not re-spend ~7.6h re-deriving it.
RUN_STEP1_MATCHED_BUDGET_LODO = False
print('RUN_STEP1_MATCHED_BUDGET_LODO:', RUN_STEP1_MATCHED_BUDGET_LODO)


# In[14]:


REQUIREMENTS_TEXT = '''tensorflow>=2.16
tensorflow-hub
numpy
pandas
scikit-learn
matplotlib
Pillow
mlflow
'''

REQUIREMENTS_PATH = PROJECT_DIR / 'requirements_stitchnet_q1_ready.txt'
REQUIREMENTS_PATH.write_text(REQUIREMENTS_TEXT)
print('Requirements file written:', REQUIREMENTS_PATH)
print(REQUIREMENTS_TEXT)
print('MLflow is listed for reproducibility but disabled by default to avoid Kaggle dependency conflicts.')
print('In a clean local environment, enable it by setting INSTALL_MLFLOW=True and ENABLE_MLFLOW=True.')


# In[15]:


import os
import sys
import subprocess

if INSTALL_MLFLOW:
    try:
        import mlflow
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'mlflow'])


# Round-3 R&R addendum: resolve the StitchingNet dataset via kagglehub instead
# of relying on Kaggle's /kaggle/input read-only mount (only available inside
# Kaggle-hosted notebooks). kagglehub authenticates the same way the `kaggle`
# CLI does -- either ~/.kaggle/kaggle.json (Kaggle account -> Settings -> API
# -> Create New Token) or the KAGGLE_USERNAME / KAGGLE_KEY environment
# variables -- and downloads the dataset to a local cache directory on first
# access only. Later cells/re-runs reuse that cache, so there is no manual
# `kaggle datasets download` + unzip step, and no full re-download once cached.
KAGGLE_DATASET_SLUG = 'hyungjung/stitchingnet-dataset'

try:
    import kagglehub
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])
    import kagglehub

_kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
_has_kaggle_json = _kaggle_json.exists()
_has_kaggle_env = bool(os.environ.get('KAGGLE_USERNAME')) and bool(os.environ.get('KAGGLE_KEY'))
if not (_has_kaggle_json or _has_kaggle_env):
    raise RuntimeError(
        'No Kaggle credentials found for kagglehub. Either place kaggle.json at '
        '~/.kaggle/kaggle.json (from https://www.kaggle.com/settings -> API -> '
        'Create New Token), or set the KAGGLE_USERNAME and KAGGLE_KEY environment '
        'variables before starting this JupyterLab kernel.'
    )

print('Resolving StitchingNet dataset via kagglehub (downloads to a local cache on first run only)...')
_kagglehub_path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
DATASET_ROOT = Path(_kagglehub_path)
print('kagglehub cache path:', DATASET_ROOT)


# In[16]:


import os
import json
import time
import random
import platform
import zipfile
import shutil
import gc
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import tensorflow as tf

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:
        print('Could not set GPU memory growth:', exc)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, balanced_accuracy_score, log_loss, brier_score_loss

try:
    import mlflow
    import mlflow.keras
    MLFLOW_AVAILABLE = True
except Exception as exc:
    mlflow = None
    MLFLOW_AVAILABLE = False
    print('MLflow unavailable:', exc)

try:
    import tensorflow_hub as hub
except Exception as exc:
    hub = None
    print('tensorflow_hub unavailable:', exc)

try:
    import matplotlib
    matplotlib.use('Agg')  # headless server, no display/browser attached
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    print('matplotlib unavailable:', exc)

for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def shell(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return str(exc)

def autodetect_dataset_root(preferred):
    candidates = [preferred, Path('/kaggle/input/stitchingnet-dataset')]
    input_root = Path('/kaggle/input')
    if input_root.exists():
        candidates += [path for path in input_root.rglob('*') if path.is_dir() and 'stitchingnet' in path.name.lower()]
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob('*.jpg')):
            return candidate
    return preferred

DATASET_ROOT = autodetect_dataset_root(DATASET_ROOT)

runtime_info = {
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'python': sys.version,
    'platform': platform.platform(),
    'tensorflow': tf.__version__,
    'mlflow_available': MLFLOW_AVAILABLE,
    'tf_built_with_cuda': tf.test.is_built_with_cuda(),
    'physical_gpus': [str(gpu) for gpu in tf.config.list_physical_devices('GPU')],
    'nvidia_smi': shell('nvidia-smi'),
    'dataset_root': str(DATASET_ROOT),
    'run_mode': RUN_MODE,
    'seeds': SEEDS,
    'deployment_validity_model_name': DEPLOYMENT_VALIDITY_MODEL_NAME,
    'final_selection_policy': FINAL_SELECTION_POLICY,
    'fold_batch_size': FOLD_BATCH_SIZE,
    'domain_epochs': DOMAIN_EPOCHS,
    'adapt_epochs': ADAPT_EPOCHS,
}

(PROJECT_DIR / 'runtime_info.json').write_text(json.dumps(runtime_info, indent=2))

MLFLOW_EXPERIMENT_NAME = 'stitchnet-q1-ready-deployment-validity'
if ENABLE_MLFLOW and MLFLOW_AVAILABLE:
    try:
        mlflow.set_tracking_uri(f'sqlite:///{MLFLOW_DB}')
        existing_experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if existing_experiment is None:
            mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME, artifact_location=str(MLFLOW_ARTIFACT_DIR.resolve()))
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        runtime_info['mlflow_tracking_uri'] = f'sqlite:///{MLFLOW_DB}'
        runtime_info['mlflow_artifact_dir'] = str(MLFLOW_ARTIFACT_DIR)
    except Exception as exc:
        MLFLOW_AVAILABLE = False
        runtime_info['mlflow_setup_error'] = str(exc)
        print('MLflow disabled after setup failure:', exc)
    (PROJECT_DIR / 'runtime_info.json').write_text(json.dumps(runtime_info, indent=2))

runtime_info


# In[17]:


IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
CLASS_ID_TO_NAME = {
    '0': 'Normal',
    '1': 'Skipped stitch',
    '2': 'Broken stitch',
    '3': 'Pinched fabric',
    '4': 'Crooked seam',
    '5': 'Thread sagging',
    '6': 'Puckering',
    '7': 'Stain and damage',
    '8': 'Needle mark',
    '9': 'Bobbin thread pulling up',
    '10': 'Overlapped stitch',
}

def infer_label_domain(path, root):
    stem_parts = path.stem.split('_')
    if len(stem_parts) >= 2 and stem_parts[0].isalpha() and stem_parts[1].isdigit():
        domain = stem_parts[0]
        class_id = str(int(stem_parts[1]))
        return CLASS_ID_TO_NAME.get(class_id, class_id), class_id, domain
    parent = path.parent.name
    grandparent = path.parent.parent.name if path.parent.parent != root.parent else ''
    if parent.isdigit() or parent.split('.')[0].isdigit():
        class_id = str(int(parent.split('.')[0]))
        return CLASS_ID_TO_NAME.get(class_id, parent), class_id, grandparent
    class_id = parent.split('.')[0] if parent.split('.')[0].isdigit() else parent
    domain = grandparent if grandparent else (stem_parts[0] if stem_parts else 'unknown')
    return parent, class_id, domain

image_paths = sorted([path for path in DATASET_ROOT.rglob('*') if path.suffix.lower() in IMAGE_EXTS])
records = []
for path in image_paths:
    label, class_id, domain = infer_label_domain(path, DATASET_ROOT)
    records.append({'filepath': str(path), 'filename': path.name, 'label': label, 'class_id': str(class_id), 'fabric_thread': str(domain)})

df = pd.DataFrame(records)
dataset_profile = {
    'image_count': int(len(df)),
    'class_count': int(df['class_id'].nunique()) if len(df) else 0,
    'domain_count': int(df['fabric_thread'].nunique()) if len(df) else 0,
    'class_counts': df.groupby(['class_id', 'label']).size().reset_index(name='n').to_dict('records') if len(df) else [],
    'domain_counts': df.groupby('fabric_thread').size().reset_index(name='n').to_dict('records') if len(df) else [],
}
(PROJECT_DIR / 'dataset_profile.json').write_text(json.dumps(dataset_profile, indent=2))
print(json.dumps(dataset_profile, indent=2)[:4000])
if len(df) != 14565:
    print('WARNING: scanned image count differs from expected 14565.')


# In[18]:


def make_exact_6_2_2_split(data, seed):
    strata_cols = ['fabric_thread', 'class_id']
    counts = data.groupby(strata_cols).size().reset_index(name='n')
    total = int(counts['n'].sum())
    targets = {'train': int(round(total * 0.6)), 'val': int(round(total * 0.2))}
    targets['test'] = total - targets['train'] - targets['val']
    counts['train_n'] = np.floor(counts['n'] * 0.6).astype(int)
    counts['val_n'] = np.floor(counts['n'] * 0.2).astype(int)
    counts['test_n'] = counts['n'] - counts['train_n'] - counts['val_n']
    counts['train_frac'] = counts['n'] * 0.6 - counts['train_n']
    counts['val_frac'] = counts['n'] * 0.2 - counts['val_n']
    train_extra = targets['train'] - int(counts['train_n'].sum())
    if train_extra > 0:
        idx = counts.sort_values('train_frac', ascending=False).head(train_extra).index
        counts.loc[idx, 'train_n'] += 1
        counts.loc[idx, 'test_n'] -= 1
    val_extra = targets['val'] - int(counts['val_n'].sum())
    if val_extra > 0:
        idx = counts[counts['test_n'] > 0].sort_values('val_frac', ascending=False).head(val_extra).index
        counts.loc[idx, 'val_n'] += 1
        counts.loc[idx, 'test_n'] -= 1
    pieces = []
    allocation = counts.set_index(strata_cols)[['train_n', 'val_n', 'test_n']].to_dict('index')
    for key, group in data.groupby(strata_cols, group_keys=False):
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        alloc = allocation[key]
        splits = ['train'] * int(alloc['train_n']) + ['val'] * int(alloc['val_n']) + ['test'] * int(alloc['test_n'])
        out = group.copy()
        out['split'] = splits
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True), counts, targets

split_df, allocation_df, split_targets = make_exact_6_2_2_split(df, SEEDS[0])
split_df['protocol_id'] = 'p0_6_2_2_by_fabric_thread_and_class'
split_df['seed'] = SEEDS[0]
split_df.to_csv(SPLIT_DIR / f'p0_6_2_2_by_fabric_thread_seed_{SEEDS[0]}.csv', index=False)
allocation_df.to_csv(SPLIT_DIR / f'p0_stratum_allocation_seed_{SEEDS[0]}.csv', index=False)
count_summary = {
    'total_images': int(len(split_df)),
    'split_counts': split_df['split'].value_counts().to_dict(),
    'class_counts': split_df.groupby(['class_id', 'split']).size().unstack(fill_value=0).to_dict(),
    'fabric_thread_counts': split_df.groupby(['fabric_thread', 'split']).size().unstack(fill_value=0).to_dict(),
}
(SPLIT_DIR / f'p0_split_count_summary_seed_{SEEDS[0]}.json').write_text(json.dumps(count_summary, indent=2))
print('split targets:', split_targets)
print(split_df['split'].value_counts().to_dict())


# In[19]:


label_order = sorted(split_df['class_id'].unique(), key=lambda value: int(value) if str(value).isdigit() else str(value))
label_name_by_id = split_df.drop_duplicates('class_id').set_index('class_id')['label'].to_dict()
class_names = [label_name_by_id[class_id] for class_id in label_order]


def frame_for_split(data, split_name):
    out = data[data['split'] == split_name].copy()
    out['class_id'] = out['class_id'].astype(str)
    return out


train_df = frame_for_split(split_df, 'train')
val_df = frame_for_split(split_df, 'val')
test_df = frame_for_split(split_df, 'test')


def make_generators(train_part, val_part, test_part, batch_size=BATCH_SIZE, shuffle_train=True, seed=PRIMARY_SEED):
    train_part = train_part.copy()
    val_part = val_part.copy()
    test_part = test_part.copy()
    for part in [train_part, val_part, test_part]:
        part['class_id'] = part['class_id'].astype(str)
    train_datagen = ImageDataGenerator(rescale=1./255)
    eval_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_dataframe(train_part, x_col='filepath', y_col='class_id', classes=label_order, target_size=IMG_SIZE, batch_size=batch_size, class_mode='categorical', shuffle=shuffle_train, seed=seed)
    val_gen = eval_datagen.flow_from_dataframe(val_part, x_col='filepath', y_col='class_id', classes=label_order, target_size=IMG_SIZE, batch_size=batch_size, class_mode='categorical', shuffle=False)
    test_gen = eval_datagen.flow_from_dataframe(test_part, x_col='filepath', y_col='class_id', classes=label_order, target_size=IMG_SIZE, batch_size=batch_size, class_mode='categorical', shuffle=False)
    return train_gen, val_gen, test_gen


train_gen, val_gen, test_gen = make_generators(train_df, val_df, test_df, seed=PRIMARY_SEED)


# In[20]:


CNN_MODEL_ALIASES = {
    'DenseNet121': 'DenseNet121',
    'InceptionV3': 'InceptionV3',
    'MobileNetV1': 'MobileNet',
    'ResNet152V2': 'ResNet152V2',
    'VGG19': 'VGG19',
    'Xception': 'Xception',
}
HEAD_LAYER_NAMES = {'global_average_pooling', 'head_dropout', 'compact_bottleneck', 'class_output'}
STRATEGIES = {'S.C': [False, False, True, True, True]}


def build_transfer_model(model_name, trainable=False, num_classes=11):
    app = getattr(tf.keras.applications, CNN_MODEL_ALIASES[model_name])
    base = app(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base.trainable = trainable
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling')(base.output)
    x = tf.keras.layers.Dropout(HEAD_DROPOUT, name='head_dropout')(x)
    if HEAD_DENSE_UNITS and int(HEAD_DENSE_UNITS) > 0:
        x = tf.keras.layers.Dense(int(HEAD_DENSE_UNITS), activation='relu', name='compact_bottleneck')(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='class_output')(x)
    return tf.keras.Model(inputs=base.input, outputs=output, name=model_name.replace('-', '_'))


def section_for_layer(model_name, layer_name):
    if layer_name in HEAD_LAYER_NAMES:
        return None
    if model_name == 'DenseNet121':
        if layer_name.startswith(('conv1', 'pool1')): return 0
        if layer_name.startswith(('conv2', 'pool2')): return 1
        if layer_name.startswith(('conv3', 'pool3')): return 2
        if layer_name.startswith(('conv4', 'pool4')): return 3
        return 4
    if model_name == 'ResNet152V2':
        if layer_name.startswith(('conv1', 'pool1')): return 0
        if layer_name.startswith('conv2'): return 1
        if layer_name.startswith('conv3'): return 2
        if layer_name.startswith('conv4'): return 3
        return 4
    if model_name == 'Xception':
        if layer_name.startswith(('block1', 'block2')): return 0
        if layer_name.startswith('block3'): return 1
        if layer_name.startswith('block4'): return 2
        if any(layer_name.startswith(f'block{i}') for i in range(5, 14)): return 3
        return 4
    if model_name == 'MobileNetV1':
        if layer_name.startswith('conv1'): return 0
        import re
        match = re.match(r'conv_(?:dw|pw)_(\d+)', layer_name)
        if not match: return 0
        block = int(match.group(1))
        if block <= 2: return 1
        if block <= 4: return 2
        if block <= 8: return 3
        return 4
    if model_name == 'VGG19':
        if layer_name.startswith('block1'): return 0
        if layer_name.startswith('block2'): return 1
        if layer_name.startswith('block3'): return 2
        if layer_name.startswith('block4'): return 3
        return 4
    if model_name == 'InceptionV3':
        if not layer_name.startswith('mixed'): return 0
        import re
        match = re.match(r'mixed(\d+)', layer_name)
        if not match: return 0
        block = int(match.group(1))
        if block <= 2: return 1
        if block <= 5: return 2
        if block <= 8: return 3
        return 4
    return None


def apply_strategy(model, model_name, strategy_flags):
    for layer in model.layers:
        section = section_for_layer(model_name, layer.name)
        if section is None:
            layer.trainable = layer.name in HEAD_LAYER_NAMES
        else:
            layer.trainable = bool(strategy_flags[section])
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    rows = []
    for layer in model.layers:
        section = section_for_layer(model_name, layer.name)
        rows.append({'layer': layer.name, 'section': 'head' if layer.name in HEAD_LAYER_NAMES else (['I','II','III','IV','V'][section] if section is not None else 'unmapped'), 'trainable': bool(layer.trainable)})
    return pd.DataFrame(rows)


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def count_params(weights):
    return int(np.sum([np.prod(weight.shape) for weight in weights]))


def model_size_mb(path):
    path = Path(path)
    if not path.exists(): return None
    if path.is_file(): return path.stat().st_size / (1024 ** 2)
    return sum(child.stat().st_size for child in path.rglob('*') if child.is_file()) / (1024 ** 2)


# In[21]:


def evaluate_model(model, generator, model_name):
    generator.reset()
    y_true = generator.classes
    predictions = []
    probabilities = []
    batch_times = []
    if len(generator) > 0:
        warm_x, _ = generator[0]
        model.predict(warm_x, verbose=0)
    for i in range(len(generator)):
        x_batch, _ = generator[i]
        start = time.time()
        pred = model.predict(x_batch, verbose=0)
        batch_times.append(time.time() - start)
        probabilities.append(pred)
        predictions.extend(np.argmax(pred, axis=1).tolist())
    y_prob = np.vstack(probabilities)[:len(y_true)] if probabilities else np.zeros((0, len(label_order)))
    y_pred = np.array(predictions[:len(y_true)])
    total_predict_time = float(np.sum(batch_times))
    avg_image_s = total_predict_time / max(len(y_true), 1)
    batch_times_np = np.array(batch_times, dtype=float) if batch_times else np.array([np.nan])
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'inference_s_per_batch': total_predict_time / max(len(generator), 1),
        'inference_s_per_image': avg_image_s,
        'inference_ms_per_image': avg_image_s * 1000,
        'fps_images_per_second': len(y_true) / total_predict_time if total_predict_time > 0 else None,
        'batch_latency_p50_s': float(np.nanpercentile(batch_times_np, 50)),
        'batch_latency_p95_s': float(np.nanpercentile(batch_times_np, 95)),
        'batch_latency_p99_s': float(np.nanpercentile(batch_times_np, 99)),
        'num_test_images': int(len(y_true)),
        'num_test_batches': int(len(generator)),
        'batch_size': int(generator.batch_size),
        'total_params': count_params(model.weights),
        'trainable_params': count_params(model.trainable_weights),
        'predicted_classes': int(len(np.unique(y_pred))) if len(y_pred) else 0,
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_order))))
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    return metrics, cm, report, y_true, y_pred, y_prob


def save_eval_outputs(run_name, metrics, cm, report, prefix):
    safe = run_name.replace('/', '_').replace(':', '_')
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / f'{prefix}_{safe}_metrics.csv', index=False)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(RESULTS_DIR / f'{prefix}_{safe}_confusion_matrix.csv')
    (RESULTS_DIR / f'{prefix}_{safe}_classification_report.json').write_text(json.dumps(report, indent=2))


TRACKING_ROWS = []


def log_run(run_name, params, metrics, artifacts=None):
    artifacts = artifacts or []
    row = {
        'run_name': run_name,
        **{f'param_{k}': v for k, v in params.items()},
        **{f'metric_{k}': v for k, v in metrics.items() if isinstance(v, (int, float, str, bool, type(None)))}
    }
    TRACKING_ROWS.append(row)
    pd.DataFrame(TRACKING_ROWS).to_csv(RESULTS_DIR / 'tracking_runs.csv', index=False)
    if ENABLE_MLFLOW and MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and np.isfinite(v)}
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics)
            for artifact in artifacts:
                artifact = Path(artifact)
                if artifact.exists():
                    mlflow.log_artifact(str(artifact))


# In[22]:


def release_keras_memory(*objects):
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    tf.keras.backend.clear_session()


def train_head_only(model_name, train_gen, val_gen, test_gen):
    tf.keras.backend.clear_session()
    run_name = f'{model_name}-wide-head'
    model = compile_model(build_transfer_model(model_name, trainable=False, num_classes=len(label_order)), HEAD_LR)
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.3, min_lr=1e-6)]
    start = time.time()
    history = model.fit(train_gen, validation_data=val_gen, epochs=HEAD_EPOCHS, callbacks=callbacks, verbose=1)
    train_seconds = time.time() - start
    metrics, cm, report, _, _, _ = evaluate_model(model, test_gen, model_name)
    model_path = MODEL_DIR / f'wide_{model_name}.keras'
    if SAVE_INTERMEDIATE_KERAS_MODELS:
        model.save(model_path)
    metrics.update({'model_size_mb': model_size_mb(model_path), 'stage': 'wide_screening_head_only', 'train_seconds': train_seconds, 'best_epoch': int(np.argmin(history.history.get('val_loss', [0])) + 1)})
    save_eval_outputs(model_name, metrics, cm, report, 'wide')
    log_run(run_name, {'model': model_name, 'stage': 'wide', 'head_epochs': HEAD_EPOCHS, 'lr': HEAD_LR}, metrics, [model_path])
    return model, metrics


def train_stable_sc(
    model_name,
    train_gen,
    val_gen,
    test_gen,
    run_suffix='S.C.stable',
    save_prefix='fine',
    head_epochs=None,
    finetune_epochs=None,
    head_lr=HEAD_LR,
    finetune_lr=FINETUNE_LR,
    save_predictions=True,
    save_model=False,
):
    tf.keras.backend.clear_session()
    head_epochs = HEAD_EPOCHS if head_epochs is None else int(head_epochs)
    finetune_epochs = FINETUNE_EPOCHS if finetune_epochs is None else int(finetune_epochs)
    run_name = f'{model_name}-{run_suffix}'
    model = compile_model(build_transfer_model(model_name, trainable=False, num_classes=len(label_order)), head_lr)
    head_callbacks = [EarlyStopping(monitor='val_loss', patience=max(2, min(5, head_epochs // 3)), restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.3, min_lr=1e-6)]
    start = time.time()
    head_history = model.fit(train_gen, validation_data=val_gen, epochs=head_epochs, callbacks=head_callbacks, verbose=1)
    section_map = apply_strategy(model, model_name, STRATEGIES['S.C'])
    section_map_path = RESULTS_DIR / f'section_map_{run_name}.csv'
    section_map.to_csv(section_map_path, index=False)
    model = compile_model(model, finetune_lr)
    ft_callbacks = [EarlyStopping(monitor='val_loss', patience=max(2, min(5, finetune_epochs // 2)), restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.3, min_lr=1e-7)]
    ft_history = model.fit(train_gen, validation_data=val_gen, epochs=finetune_epochs, callbacks=ft_callbacks, verbose=1)
    train_seconds = time.time() - start
    metrics, cm, report, y_true, y_pred, y_prob = evaluate_model(model, test_gen, run_name)
    model_path = MODEL_DIR / f'{save_prefix}_{run_name}.keras'
    if save_model or SAVE_INTERMEDIATE_KERAS_MODELS:
        model.save(model_path)
    metrics.update({
        'model_size_mb': model_size_mb(model_path),
        'stage': 'stable_two_stage_sc',
        'strategy': run_suffix,
        'train_seconds': train_seconds,
        'head_epochs_requested': head_epochs,
        'finetune_epochs_requested': finetune_epochs,
        'head_best_epoch': int(np.argmin(head_history.history.get('val_loss', [0])) + 1),
        'finetune_best_epoch': int(np.argmin(ft_history.history.get('val_loss', [0])) + 1),
    })
    save_eval_outputs(run_name, metrics, cm, report, save_prefix)
    if save_predictions:
        np.save(RESULTS_DIR / f'{save_prefix}_{run_name}_y_true.npy', y_true)
        np.save(RESULTS_DIR / f'{save_prefix}_{run_name}_y_pred.npy', y_pred)
        np.save(RESULTS_DIR / f'{save_prefix}_{run_name}_y_prob.npy', y_prob)
    log_run(run_name, {'model': model_name, 'stage': 'fine_sc_stable', 'head_lr': head_lr, 'finetune_lr': finetune_lr, 'head_epochs': head_epochs, 'finetune_epochs': finetune_epochs}, metrics, [model_path, section_map_path])
    return model, metrics, (y_true, y_pred, y_prob)


# In[23]:


if RUN_FULL_ORIGINAL_PIPELINE:
    wide_results = []
    fine_results = []
    fine_models = {}
    fine_predictions = {}

    for model_name in PHASE0_CNN_MODELS:
        try:
            model, metrics = train_head_only(model_name, train_gen, val_gen, test_gen)
            wide_results.append(metrics)
        except Exception as exc:
            wide_results.append({'model': model_name, 'stage': 'wide_screening_head_only', 'error': str(exc)})
        finally:
            release_keras_memory(locals().get('model'))

    for model_name in PHASE0_CNN_MODELS:
        try:
            keep_model = model_name == DEPLOYMENT_VALIDITY_MODEL_NAME
            model, metrics, prediction_bundle = train_stable_sc(
                model_name,
                train_gen,
                val_gen,
                test_gen,
                save_model=keep_model,
            )
            if keep_model:
                fine_models[model_name] = model
                fine_predictions[model_name] = prediction_bundle
            else:
                release_keras_memory(model)
            fine_results.append(metrics)
        except Exception as exc:
            fine_results.append({'model': f'{model_name}-S.C.stable', 'stage': 'stable_two_stage_sc', 'strategy': 'S.C.stable', 'error': str(exc)})

    wide_results_df = pd.DataFrame(wide_results)
    fine_results_df = pd.DataFrame(fine_results)
    wide_results_df.to_csv(RESULTS_DIR / 'wide_screening_results.csv', index=False)
    fine_results_df.to_csv(RESULTS_DIR / 'systematic_fine_tuning_results.csv', index=False)
    display(wide_results_df)
    display(fine_results_df.sort_values('f1_weighted', ascending=False, na_position='last') if 'f1_weighted' in fine_results_df else fine_results_df)
else:
    print('Skipped (RUN_FULL_ORIGINAL_PIPELINE=False): backbone wide-screening + fine-tuning sweep (PHASE0_CNN_MODELS)')


# In[24]:


if RUN_FULL_ORIGINAL_PIPELINE:
    vit_rows = []
    try:
        if hub is None:
            raise RuntimeError('tensorflow_hub is unavailable')
        vit_url = 'https://www.kaggle.com/models/spsayakpaul/vision-transformer/TensorFlow2/vit-l16-fe/1'
        fe_layer = hub.KerasLayer(vit_url, trainable=False, name='vit_feature_extractor')
        inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        try:
            outputs = fe_layer(inputs)
            vit_rows.append({'model': 'vit-l16-fe', 'status': 'symbolic_call_pass', 'detail': str(outputs)[:1000]})
        except Exception as exc:
            vit_rows.append({'model': 'vit-l16-fe', 'status': 'symbolic_call_fail', 'detail': str(exc)[:1000]})
    except Exception as exc:
        vit_rows.append({'model': 'vit-l16-fe', 'status': 'setup_fail', 'detail': str(exc)[:1000]})
    vit_status_df = pd.DataFrame(vit_rows)
    vit_status_df.to_csv(RESULTS_DIR / 'vit_l16_status.csv', index=False)
    display(vit_status_df)
else:
    print('Skipped (RUN_FULL_ORIGINAL_PIPELINE=False): ViT-L/16 feature-extractor probe')


# In[25]:


if RUN_FULL_ORIGINAL_PIPELINE:
    valid_fine = fine_results_df.dropna(subset=['f1_weighted']) if 'f1_weighted' in fine_results_df else pd.DataFrame()
    if len(valid_fine) and FINAL_SELECTION_POLICY == 'deployment_validity_backbone':
        deployment_rows = valid_fine[valid_fine['model'].astype(str).str.startswith(DEPLOYMENT_VALIDITY_MODEL_NAME)]
        if len(deployment_rows):
            final_row = deployment_rows.sort_values(['f1_weighted', 'inference_ms_per_image'], ascending=[False, True]).iloc[0]
        else:
            final_row = valid_fine.sort_values(['f1_weighted', 'inference_ms_per_image'], ascending=[False, True]).iloc[0]
    elif len(valid_fine):
        final_row = valid_fine.sort_values(['f1_weighted', 'inference_ms_per_image'], ascending=[False, True]).iloc[0]
    else:
        final_row = pd.Series({'model': f'{DEPLOYMENT_VALIDITY_MODEL_NAME}-S.C.stable'})

    final_model_label = str(final_row['model'])
    final_base_model = final_model_label.split('-')[0]
    final_model = fine_models.get(final_base_model)
    if final_model is None:
        candidate = MODEL_DIR / f'fine_{final_model_label}.keras'
        if candidate.exists():
            final_model = tf.keras.models.load_model(candidate, compile=False)
            final_model = compile_model(final_model, FINETUNE_LR)

    final_keras_path = MODEL_DIR / 'final_model.keras'
    if final_model is not None and SAVE_FINAL_KERAS_MODEL:
        final_model.save(final_keras_path)

    final_selection = {
        'final_model_label': final_model_label,
        'final_base_model': final_base_model,
        'selection_policy': FINAL_SELECTION_POLICY,
        'deployment_validity_model_name': DEPLOYMENT_VALIDITY_MODEL_NAME,
        'final_model_keras_path': str(final_keras_path) if final_keras_path.exists() else '',
    }
    (PROJECT_DIR / 'final_model_selection.json').write_text(json.dumps(final_selection, indent=2))
    print(final_selection)
else:
    print('Skipped (RUN_FULL_ORIGINAL_PIPELINE=False): final in-distribution model selection / final_model.keras save')


# In[ ]:


def make_lodo_split(source_df, held_out_domain, seed):
    target = source_df[source_df['fabric_thread'].astype(str) == str(held_out_domain)].copy()
    source = source_df[source_df['fabric_thread'].astype(str) != str(held_out_domain)].copy()
    source = source.assign(_rand=np.random.default_rng(seed).random(len(source))).sort_values(['class_id', '_rand']).drop(columns='_rand')
    source['split'] = 'train'
    val_idx = []
    for _, group in source.groupby('class_id', sort=True):
        val_idx.extend(group.index[:max(1, int(round(len(group) * 0.2)))].tolist())
    source.loc[val_idx, 'split'] = 'val'
    target['split'] = 'test'
    out = pd.concat([source, target], ignore_index=True)
    out['protocol_id'] = 'p1_leave_one_fabric_thread_domain_out'
    out['held_out_domain'] = held_out_domain
    out['seed'] = seed
    return out


def write_rows(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def run_domain_generalisation(model_name):
    rows = []
    out_path = RESULTS_DIR / 'domain_generalisation_results.csv'
    if out_path.exists():
        out_path.unlink()
    domains = sorted(split_df['fabric_thread'].astype(str).unique())
    for seed in DEPLOYMENT_PROTOCOL_SEEDS:
        for domain in domains:
            release_keras_memory()
            fold_df = make_lodo_split(df, domain, seed)
            fold_path = SPLIT_DIR / f'p1_lodo_domain_{domain}_seed_{seed}.csv'
            fold_df.to_csv(fold_path, index=False)
            tr = fold_df[fold_df['split'] == 'train'].copy()
            va = fold_df[fold_df['split'] == 'val'].copy()
            te = fold_df[fold_df['split'] == 'test'].copy()
            tr_gen, va_gen, te_gen = make_generators(tr, va, te, batch_size=FOLD_BATCH_SIZE, seed=seed)
            try:
                model, metrics, _ = train_stable_sc(
                    model_name,
                    tr_gen,
                    va_gen,
                    te_gen,
                    run_suffix=f'S.C.lodo.{domain}.seed{seed}',
                    save_prefix='domain',
                    head_epochs=DOMAIN_EPOCHS,
                    finetune_epochs=DOMAIN_EPOCHS,
                    finetune_lr=DOMAIN_LR,
                    save_predictions=False,
                    save_model=False,
                )
                metrics.update({'held_out_domain': domain, 'seed': seed, 'protocol_id': 'p1_leave_one_fabric_thread_domain_out', 'error': np.nan})
                release_keras_memory(model)
            except Exception as exc:
                metrics = {'model': f'{model_name}-S.C.lodo.{domain}.seed{seed}', 'held_out_domain': domain, 'seed': seed, 'protocol_id': 'p1_leave_one_fabric_thread_domain_out', 'error': str(exc)}
            rows.append(metrics)
            write_rows(out_path, rows)
    return pd.DataFrame(rows)


if RUN_FULL_ORIGINAL_PIPELINE:
    domain_results_df = run_domain_generalisation(DEPLOYMENT_VALIDITY_MODEL_NAME)
    within_rows = valid_fine[valid_fine['model'].astype(str).str.startswith(DEPLOYMENT_VALIDITY_MODEL_NAME)] if len(valid_fine) and 'f1_macro' in valid_fine else pd.DataFrame()
    within_f1 = float(within_rows['f1_macro'].max()) if len(within_rows) else np.nan
    if 'f1_macro' in domain_results_df:
        domain_results_df['within_distribution_macro_f1'] = within_f1
        domain_results_df['domain_performance_gap'] = within_f1 - domain_results_df['f1_macro']
        domain_results_df.to_csv(RESULTS_DIR / 'domain_generalisation_results.csv', index=False)
    display(domain_results_df)
else:
    print('Skipped (RUN_FULL_ORIGINAL_PIPELINE=False): original zero-effort 33-run LODO sweep (already in submitted evidence package). make_lodo_split/run_domain_generalisation remain defined for reuse below.')


# In[ ]:


def make_cold_start_split(source_df, held_out_domain, support_size, seed, adaptation_val_per_class):
    target = source_df[source_df['fabric_thread'].astype(str) == str(held_out_domain)].copy()
    source = source_df[source_df['fabric_thread'].astype(str) != str(held_out_domain)].copy()
    source['split'] = 'source_train'
    rng = np.random.default_rng(seed)
    parts = [source]
    for _, group in target.groupby('class_id', sort=True):
        shuffled = group.sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1))).copy()
        needed = support_size + adaptation_val_per_class + 1
        if len(shuffled) < needed:
            return None
        support = shuffled.iloc[:support_size].copy()
        adap_val = shuffled.iloc[support_size:support_size + adaptation_val_per_class].copy()
        final_test = shuffled.iloc[support_size + adaptation_val_per_class:].copy()
        support['split'] = 'target_support'
        adap_val['split'] = 'target_adaptation_val'
        final_test['split'] = 'target_final_test'
        parts.extend([support, adap_val, final_test])
    out = pd.concat(parts, ignore_index=True)
    out['protocol_id'] = 'p2_cold_start_adaptation'
    out['held_out_domain'] = held_out_domain
    out['support_per_class'] = support_size
    out['adaptation_val_per_class'] = adaptation_val_per_class
    out['seed'] = seed
    return out


def run_cold_start(model_name):
    rows = []
    out_path = RESULTS_DIR / 'cold_start_adaptation_results.csv'
    if out_path.exists():
        out_path.unlink()
    domains = sorted(split_df['fabric_thread'].astype(str).unique())
    for seed in DEPLOYMENT_PROTOCOL_SEEDS:
        for domain in domains:
            for support_size in SUPPORT_SIZES:
                release_keras_memory()
                fold_df = make_cold_start_split(df, domain, support_size, seed, ADAPTATION_VAL_PER_CLASS)
                if fold_df is None:
                    metrics = {'model': model_name, 'held_out_domain': domain, 'support_per_class': support_size, 'seed': seed, 'protocol_id': 'p2_cold_start_adaptation', 'error': 'insufficient_target_samples'}
                    rows.append(metrics)
                    write_rows(out_path, rows)
                    continue
                fold_path = SPLIT_DIR / f'p2_cold_start_domain_{domain}_support_{support_size}_seed_{seed}.csv'
                fold_df.to_csv(fold_path, index=False)
                source_train = fold_df[fold_df['split'] == 'source_train'].copy()
                support = fold_df[fold_df['split'] == 'target_support'].copy()
                adap_val = fold_df[fold_df['split'] == 'target_adaptation_val'].copy()
                final_test = fold_df[fold_df['split'] == 'target_final_test'].copy()
                train_part = pd.concat([source_train, support], ignore_index=True) if support_size > 0 else source_train
                tr_gen, va_gen, te_gen = make_generators(train_part, adap_val, final_test, batch_size=FOLD_BATCH_SIZE, seed=seed)
                try:
                    model, metrics, _ = train_stable_sc(
                        model_name,
                        tr_gen,
                        va_gen,
                        te_gen,
                        run_suffix=f'S.C.cold.{domain}.{support_size}.seed{seed}',
                        save_prefix='cold',
                        head_epochs=ADAPT_EPOCHS,
                        finetune_epochs=ADAPT_EPOCHS,
                        finetune_lr=ADAPT_LR,
                        save_predictions=False,
                        save_model=False,
                    )
                    metrics.update({'held_out_domain': domain, 'support_per_class': support_size, 'seed': seed, 'protocol_id': 'p2_cold_start_adaptation', 'error': np.nan})
                    release_keras_memory(model)
                except Exception as exc:
                    metrics = {'model': f'{model_name}-S.C.cold.{domain}.{support_size}.seed{seed}', 'held_out_domain': domain, 'support_per_class': support_size, 'seed': seed, 'protocol_id': 'p2_cold_start_adaptation', 'error': str(exc)}
                rows.append(metrics)
                write_rows(out_path, rows)
    return pd.DataFrame(rows)


if RUN_FULL_ORIGINAL_PIPELINE:
    cold_start_results_df = run_cold_start(DEPLOYMENT_VALIDITY_MODEL_NAME)
    cold_start_results_df.to_csv(RESULTS_DIR / 'cold_start_adaptation_results.csv', index=False)
    if 'f1_macro' in cold_start_results_df:
        cold_start_summary_df = cold_start_results_df.groupby(['support_per_class'], dropna=False).agg(
            accuracy_mean=('accuracy', 'mean'),
            accuracy_std=('accuracy', 'std'),
            f1_macro_mean=('f1_macro', 'mean'),
            f1_macro_std=('f1_macro', 'std'),
            f1_weighted_mean=('f1_weighted', 'mean'),
            f1_weighted_std=('f1_weighted', 'std'),
            n=('model', 'count'),
        ).reset_index()
        cold_start_summary_df.to_csv(RESULTS_DIR / 'cold_start_support_summary.csv', index=False)
        display(cold_start_summary_df)
    display(cold_start_results_df)
else:
    print('Skipped (RUN_FULL_ORIGINAL_PIPELINE=False): original 198-run cold-start sweep (already in submitted evidence package). make_cold_start_split/run_cold_start remain defined for reuse below.')


# In[ ]:


if RUN_FULL_ORIGINAL_PIPELINE:
    import gc

    gc.collect()
    final_keras_path = MODEL_DIR / 'final_model.keras'

    if 'final_model' in globals() and final_model is not None:
        print('Reusing in-memory final model for robustness/calibration/export:', final_model_label)
    elif final_keras_path.exists():
        release_keras_memory()
        with tf.device('/CPU:0'):
            final_model = tf.keras.models.load_model(final_keras_path, compile=False)
            final_model = compile_model(final_model, FINETUNE_LR)
        print('Reloaded final model on CPU for robustness/calibration/export:', final_keras_path)
    else:
        print('WARNING: final_model.keras not found and no in-memory final_model is available.')
else:
    print('Skipped (RUN_FULL_ORIGINAL_PIPELINE=False): final_model reload for robustness/calibration/edge/Grad-CAM')


# In[ ]:


if RUN_FULL_ORIGINAL_PIPELINE:
    def load_image_batch(paths):
        arrays = []
        for path in paths:
            img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
            arr = tf.keras.utils.img_to_array(img) / 255.0
            arrays.append(arr)
        return np.asarray(arrays, dtype='float32')

    def perturb_batch(batch, perturbation):
        x = tf.convert_to_tensor(batch, dtype=tf.float32)
        if perturbation == 'clean':
            y = x
        elif perturbation == 'brightness_down':
            y = tf.image.adjust_brightness(x, -0.20)
        elif perturbation == 'brightness_up':
            y = tf.image.adjust_brightness(x, 0.20)
        elif perturbation == 'contrast_down':
            y = tf.image.adjust_contrast(x, 0.70)
        elif perturbation == 'contrast_up':
            y = tf.image.adjust_contrast(x, 1.30)
        elif perturbation == 'gaussian_noise':
            y = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.04, seed=SEEDS[0])
        elif perturbation == 'translate':
            y = tf.roll(x, shift=6, axis=2)
        elif perturbation == 'low_resolution':
            small = tf.image.resize(x, (112, 112), method='bilinear')
            y = tf.image.resize(small, IMG_SIZE, method='bilinear')
        elif perturbation == 'gaussian_blur':
            y = tf.nn.avg_pool2d(x, ksize=3, strides=1, padding='SAME')
        else:
            y = x
        return tf.clip_by_value(y, 0.0, 1.0).numpy()

    def predict_dataframe(model, data_part, perturbation='clean', batch_size=1):
        y_true = data_part['class_id'].astype(str).map({class_id: idx for idx, class_id in enumerate(label_order)}).to_numpy()
        probs = []
        for start in range(0, len(data_part), batch_size):
            paths = data_part.iloc[start:start + batch_size]['filepath'].tolist()
            batch = load_image_batch(paths)
            batch = perturb_batch(batch, perturbation)
            probs.append(model.predict(batch, verbose=0))
        y_prob = np.vstack(probs) if probs else np.zeros((0, len(label_order)))
        y_pred = np.argmax(y_prob, axis=1) if len(y_prob) else np.array([])
        return y_true, y_pred, y_prob

    robust_rows = []
    for perturbation in ROBUSTNESS_PERTURBATIONS:
        y_true, y_pred, _ = predict_dataframe(final_model, test_df, perturbation=perturbation)
        robust_rows.append({'model': final_model_label, 'perturbation': perturbation, 'accuracy': accuracy_score(y_true, y_pred), 'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0), 'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)})
    robustness_results_df = pd.DataFrame(robust_rows)
    clean_f1 = float(robustness_results_df[robustness_results_df['perturbation'] == 'clean']['macro_f1'].iloc[0])
    robustness_results_df['macro_f1_drop_from_clean'] = clean_f1 - robustness_results_df['macro_f1']
    robustness_results_df.to_csv(RESULTS_DIR / 'robustness_results.csv', index=False)
    display(robustness_results_df)
else:
    print('Skipped (RUN_FULL_ORIGINAL_PIPELINE=False): robustness-under-perturbation evaluation')


# In[ ]:


if RUN_FULL_ORIGINAL_PIPELINE:
    def expected_calibration_error(y_true, y_prob, n_bins=15):
        confidence = np.max(y_prob, axis=1)
        prediction = np.argmax(y_prob, axis=1)
        correct = (prediction == y_true).astype(float)
        ece = 0.0
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        rows = []
        for left, right in zip(bins[:-1], bins[1:]):
            mask = (confidence > left) & (confidence <= right) if right < 1 else (confidence > left) & (confidence <= right)
            if not np.any(mask):
                rows.append({'bin_left': left, 'bin_right': right, 'count': 0, 'accuracy': np.nan, 'confidence': np.nan})
                continue
            bin_acc = float(np.mean(correct[mask]))
            bin_conf = float(np.mean(confidence[mask]))
            weight = float(np.mean(mask))
            ece += weight * abs(bin_acc - bin_conf)
            rows.append({'bin_left': left, 'bin_right': right, 'count': int(mask.sum()), 'accuracy': bin_acc, 'confidence': bin_conf})
        return float(ece), pd.DataFrame(rows)

    y_val_true, _, y_val_prob = predict_dataframe(final_model, val_df, perturbation='clean')
    y_test_true, y_test_pred, y_test_prob = predict_dataframe(final_model, test_df, perturbation='clean')
    ece, calibration_bins_df = expected_calibration_error(y_test_true, y_test_prob)
    try:
        nll = log_loss(y_test_true, y_test_prob, labels=list(range(len(label_order))))
    except Exception:
        nll = np.nan
    one_hot = np.eye(len(label_order))[y_test_true]
    brier = float(np.mean(np.sum((y_test_prob - one_hot) ** 2, axis=1)))
    confidence = np.max(y_test_prob, axis=1)
    correct = (y_test_pred == y_test_true)
    high_conf_error_rate = float(np.mean((confidence >= 0.90) & (~correct)))
    calibration_results_df = pd.DataFrame([{'model': final_model_label, 'ece_15_bins': ece, 'negative_log_likelihood': nll, 'brier_score_multiclass': brier, 'high_confidence_error_rate_0_90': high_conf_error_rate, 'test_accuracy': accuracy_score(y_test_true, y_test_pred), 'test_macro_f1': f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)}])
    coverage_rows = []
    for threshold in np.linspace(0.0, 0.99, 20):
        covered = confidence >= threshold
        coverage = float(np.mean(covered))
        risk = float(np.mean(y_test_pred[covered] != y_test_true[covered])) if np.any(covered) else np.nan
        coverage_rows.append({'model': final_model_label, 'confidence_threshold': float(threshold), 'coverage': coverage, 'risk': risk})
    risk_coverage_df = pd.DataFrame(coverage_rows)
    calibration_results_df.to_csv(RESULTS_DIR / 'calibration_results.csv', index=False)
    risk_coverage_df.to_csv(RESULTS_DIR / 'risk_coverage_results.csv', index=False)
    calibration_bins_df.to_csv(RESULTS_DIR / 'calibration_bins.csv', index=False)
    display(calibration_results_df)
    display(risk_coverage_df.head())
else:
    print('Skipped (RUN_FULL_ORIGINAL_PIPELINE=False): in-distribution calibration (ECE/NLL/Brier/risk-coverage)')


# In[ ]:


if RUN_FULL_ORIGINAL_PIPELINE:
    import os
    import subprocess
    import sys
    import pandas as pd
    import numpy as np

    export_source = MODEL_DIR / 'final_model.keras'
    if not export_source.exists():
        export_source = MODEL_DIR / f'fine_{final_model_label}.keras'

    tflite_path = MODEL_DIR / 'final_model.tflite'
    export_script = PROJECT_DIR / 'export_tflite_cpu_only.py'

    script = f"""
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf

    model = tf.keras.models.load_model({str(export_source)!r}, compile=False)

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_enable_resource_variables = True
        tflite_model = converter.convert()
        status = 'success_cpu_subprocess_dynamic_range'
    except Exception as first_exc:
        print('Quantized export failed, retrying float32:', first_exc)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.experimental_enable_resource_variables = True
        tflite_model = converter.convert()
        status = 'success_cpu_subprocess_float32'

    open({str(tflite_path)!r}, 'wb').write(tflite_model)
    print(status)
    """

    export_script.write_text(script)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '-1'

    proc = subprocess.run(
        [sys.executable, str(export_script)],
        text=True,
        capture_output=True,
        env=env,
    )

    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + "\n" + proc.stderr)

    stdout = proc.stdout.strip()
    print(stdout)
    export_status = stdout.splitlines()[-1] if stdout else 'success_cpu_subprocess'


    def run_tflite_validation(tflite_model_path, keras_model, data_part, max_images):
        subset = data_part.reset_index(drop=True).iloc[:min(max_images, len(data_part))].copy()
        y_true = subset['class_id'].astype(str).map({class_id: idx for idx, class_id in enumerate(label_order)}).to_numpy()
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_index = input_details[0]['index']
        input_dtype = input_details[0]['dtype']
        input_quant = input_details[0].get('quantization', (0.0, 0))
        try:
            interpreter.resize_tensor_input(input_index, [1, IMG_SIZE[0], IMG_SIZE[1], 3], strict=False)
        except Exception:
            pass
        interpreter.allocate_tensors()

        keras_probs = []
        tflite_probs = []
        tflite_times = []
        for path in subset['filepath'].tolist():
            batch = load_image_batch([path])
            keras_probs.append(keras_model.predict(batch, verbose=0)[0])
            x = batch.astype(input_dtype)
            if np.issubdtype(input_dtype, np.integer):
                scale, zero_point = input_quant
                if scale and scale > 0:
                    x = np.round(batch / scale + zero_point).astype(input_dtype)
            start = time.time()
            interpreter.set_tensor(input_index, x)
            interpreter.invoke()
            tflite_times.append(time.time() - start)
            out = interpreter.get_tensor(output_details[0]['index'])[0]
            out_quant = output_details[0].get('quantization', (0.0, 0))
            if np.issubdtype(out.dtype, np.integer):
                scale, zero_point = out_quant
                if scale and scale > 0:
                    out = (out.astype('float32') - zero_point) * scale
            tflite_probs.append(out.astype('float32'))

        keras_probs = np.vstack(keras_probs)
        tflite_probs = np.vstack(tflite_probs)
        keras_pred = np.argmax(keras_probs, axis=1)
        tflite_pred = np.argmax(tflite_probs, axis=1)
        return {
            'tflite_validation_status': 'pass',
            'tflite_validation_images': int(len(subset)),
            'keras_subset_accuracy': float(accuracy_score(y_true, keras_pred)),
            'tflite_subset_accuracy': float(accuracy_score(y_true, tflite_pred)),
            'keras_tflite_agreement': float(np.mean(keras_pred == tflite_pred)),
            'mean_abs_prob_diff': float(np.mean(np.abs(keras_probs - tflite_probs))),
            'max_abs_prob_diff': float(np.max(np.abs(keras_probs - tflite_probs))),
            'tflite_inference_ms_per_image': float(np.mean(tflite_times) * 1000.0),
        }

    try:
        tflite_validation = run_tflite_validation(tflite_path, final_model, test_df, TFLITE_VALIDATION_MAX_IMAGES)
    except Exception as exc:
        tflite_validation = {'tflite_validation_status': 'fail', 'tflite_validation_error': str(exc)}

    edge_device_results_df = pd.DataFrame([
        {
            'model': final_model_label,
            'artifact': 'tflite',
            'path': str(tflite_path),
            'size_mb': model_size_mb(tflite_path),
            'export_status': export_status,
            **tflite_validation,
        },
        {
            'model': final_model_label,
            'artifact': 'keras',
            'path': str(export_source),
            'size_mb': model_size_mb(export_source),
            'export_status': 'exists' if export_source.exists() else 'missing',
        },
    ])

    edge_device_results_df['hardware'] = runtime_info.get('nvidia_smi', '')[:200]
    edge_device_results_df['batch_size'] = BATCH_SIZE
    edge_device_results_df['fold_batch_size'] = FOLD_BATCH_SIZE
    edge_device_results_df['input_resolution'] = str(IMG_SIZE)

    edge_device_results_df.to_csv(RESULTS_DIR / 'edge_device_results.csv', index=False)
    pd.DataFrame([tflite_validation]).to_csv(RESULTS_DIR / 'tflite_validation_results.csv', index=False)
    display(edge_device_results_df)
else:
    print('Skipped (RUN_FULL_ORIGINAL_PIPELINE=False): TFLite export + Keras-vs-TFLite validation')


# In[ ]:


if RUN_FULL_ORIGINAL_PIPELINE:
    def find_last_conv_layer(model):
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        return None


    def make_gradcam_model(model, last_conv_layer_name):
        return tf.keras.models.Model(
            model.input,
            [model.get_layer(last_conv_layer_name).output, model.output],
        )


    def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()


    failure_rows = []
    class_to_idx = {class_id: i for i, class_id in enumerate(label_order)}

    for idx, row in test_df.reset_index(drop=True).iterrows():
        true_idx = class_to_idx[str(row['class_id'])]
        pred_idx = int(y_test_pred[idx])
        conf = float(np.max(y_test_prob[idx]))

        if pred_idx == true_idx and len([r for r in failure_rows if r['case_type'] == 'true_positive']) < 5:
            failure_rows.append({'case_type': 'true_positive', 'filepath': row['filepath'], 'true_class': true_idx, 'pred_class': pred_idx, 'confidence': conf, 'fabric_thread': row['fabric_thread']})
        if pred_idx != true_idx and len([r for r in failure_rows if r['case_type'] == 'misclassification']) < 10:
            failure_rows.append({'case_type': 'misclassification', 'filepath': row['filepath'], 'true_class': true_idx, 'pred_class': pred_idx, 'confidence': conf, 'fabric_thread': row['fabric_thread']})
        if pred_idx != true_idx and conf >= 0.90 and len([r for r in failure_rows if r['case_type'] == 'high_confidence_error']) < 10:
            failure_rows.append({'case_type': 'high_confidence_error', 'filepath': row['filepath'], 'true_class': true_idx, 'pred_class': pred_idx, 'confidence': conf, 'fabric_thread': row['fabric_thread']})

    failure_cases_df = pd.DataFrame(failure_rows)
    failure_cases_df.to_csv(RESULTS_DIR / 'failure_cases.csv', index=False)

    gradcam_status = {'status': 'not_generated', 'examples_requested': int(min(GRADCAM_MAX_EXAMPLES, len(failure_cases_df))), 'examples_generated': 0, 'error': ''}

    if plt is not None and final_model is not None and len(failure_cases_df):
        last_conv = find_last_conv_layer(final_model)
        gradcam_status['last_conv_layer'] = last_conv or ''
        if last_conv:
            try:
                grad_model = make_gradcam_model(final_model, last_conv)
                gallery_rows = failure_cases_df.head(GRADCAM_MAX_EXAMPLES)
                fig, axes = plt.subplots(len(gallery_rows), 2, figsize=(8, max(3, 3 * len(gallery_rows))))
                if len(gallery_rows) == 1:
                    axes = np.array([axes])
                generated = 0
                for ax_row, (_, sample) in zip(axes, gallery_rows.iterrows()):
                    img = tf.keras.utils.load_img(sample['filepath'], target_size=IMG_SIZE)
                    arr = tf.keras.utils.img_to_array(img) / 255.0
                    ax_row[0].imshow(img)
                    ax_row[0].axis('off')
                    ax_row[0].set_title(sample['case_type'])
                    ax_row[1].imshow(img)
                    heatmap = make_gradcam_heatmap(np.expand_dims(arr, axis=0).astype('float32'), grad_model, int(sample['pred_class']))
                    heatmap = tf.image.resize(heatmap[..., None], IMG_SIZE).numpy().squeeze()
                    ax_row[1].imshow(heatmap, cmap='jet', alpha=0.4)
                    ax_row[1].axis('off')
                    ax_row[1].set_title(f"true={sample['true_class']} pred={sample['pred_class']}")
                    generated += 1
                plt.tight_layout()
                gallery_path = FIGURE_DIR / 'failure_case_gradcam_gallery.png'
                plt.savefig(gallery_path, dpi=200)
                plt.show()
                gradcam_status.update({'status': 'pass', 'examples_generated': generated, 'path': str(gallery_path)})
            except Exception as exc:
                gradcam_status.update({'status': 'fail', 'error': str(exc)})
    else:
        gradcam_status.update({'status': 'skipped', 'error': 'matplotlib/model/failure_cases unavailable'})

    pd.DataFrame([gradcam_status]).to_csv(RESULTS_DIR / 'gradcam_status.csv', index=False)
    display(failure_cases_df.head())
    display(pd.DataFrame([gradcam_status]))
else:
    print('Skipped (RUN_FULL_ORIGINAL_PIPELINE=False): in-distribution Grad-CAM gallery + failure-case export')


# In[ ]:


if RUN_FULL_ORIGINAL_PIPELINE:
    def table_has_no_errors(path):
        if not path.exists():
            return False, 'missing'
        table = pd.read_csv(path)
        if 'error' not in table.columns:
            return True, ''
        bad = table['error'].notna() & (table['error'].astype(str).str.len() > 0) & (table['error'].astype(str).str.lower() != 'nan')
        return not bool(bad.any()), f'{int(bad.sum())} non-null error rows'


    def table_has_metrics(path, required):
        if not path.exists():
            return False, 'missing'
        table = pd.read_csv(path)
        missing = [column for column in required if column not in table.columns]
        if missing:
            return False, 'missing metrics: ' + ', '.join(missing)
        numeric_ok = table[list(required)].apply(pd.to_numeric, errors='coerce').notna().all().all()
        return bool(numeric_ok), '' if numeric_ok else 'metric columns contain non-numeric/null values'


    def expected_row_count(path, expected):
        if not path.exists():
            return False, 'missing'
        table = pd.read_csv(path)
        return len(table) == expected, f'expected {expected}, found {len(table)}'


    def audit_publication_package():
        required = [
            'runtime_info.json',
            'dataset_profile.json',
            'final_model_selection.json',
            'results/wide_screening_results.csv',
            'results/systematic_fine_tuning_results.csv',
            'results/domain_generalisation_results.csv',
            'results/cold_start_adaptation_results.csv',
            'results/robustness_results.csv',
            'results/calibration_results.csv',
            'results/risk_coverage_results.csv',
            'results/edge_device_results.csv',
            'results/tflite_validation_results.csv',
            'results/gradcam_status.csv',
            'results/failure_cases.csv',
            'splits/p0_6_2_2_by_fabric_thread_seed_0.csv',
            'models/final_model.tflite',
        ]
        rows = []
        for rel in required:
            path = PROJECT_DIR / rel
            rows.append({'artifact': rel, 'exists': path.exists(), 'size_mb': path.stat().st_size / (1024 ** 2) if path.exists() and path.is_file() else np.nan})
        audit_df = pd.DataFrame(rows)

        collapse_rows = []
        for cm_path in sorted(RESULTS_DIR.glob('*_confusion_matrix.csv')):
            cm_raw = pd.read_csv(cm_path)
            cm = cm_raw.set_index(cm_raw.columns[0])
            predicted_classes = int((cm.sum(axis=0) > 0).sum())
            true_classes = int((cm.sum(axis=1) > 0).sum())
            collapse_rows.append({'file': cm_path.name, 'predicted_classes': predicted_classes, 'true_classes': true_classes, 'status': 'PASS' if predicted_classes == true_classes else 'COLLAPSED'})
        collapse_df = pd.DataFrame(collapse_rows)

        protocol_rows = []
        protocol_specs = [
            ('domain_generalisation', RESULTS_DIR / 'domain_generalisation_results.csv', ('accuracy', 'f1_macro', 'f1_weighted'), len(DEPLOYMENT_PROTOCOL_SEEDS) * int(split_df['fabric_thread'].nunique())),
            ('cold_start_adaptation', RESULTS_DIR / 'cold_start_adaptation_results.csv', ('accuracy', 'f1_macro', 'f1_weighted'), len(DEPLOYMENT_PROTOCOL_SEEDS) * int(split_df['fabric_thread'].nunique()) * len(SUPPORT_SIZES)),
        ]
        for name, path, required_metrics, expected_rows in protocol_specs:
            no_errors, error_detail = table_has_no_errors(path)
            has_metrics, metric_detail = table_has_metrics(path, required_metrics)
            row_count_ok, row_count_detail = expected_row_count(path, expected_rows)
            protocol_rows.append({
                'protocol': name,
                'path': str(path),
                'no_errors': no_errors,
                'metric_columns_present': has_metrics,
                'row_count_ok': row_count_ok,
                'detail': '; '.join([d for d in [error_detail, metric_detail, row_count_detail if not row_count_ok else ''] if d]),
            })
        protocol_audit_df = pd.DataFrame(protocol_rows)

        tflite_df = pd.read_csv(RESULTS_DIR / 'tflite_validation_results.csv') if (RESULTS_DIR / 'tflite_validation_results.csv').exists() else pd.DataFrame()
        tflite_ok = bool(len(tflite_df) and tflite_df.iloc[0].get('tflite_validation_status') == 'pass')
        gradcam_df = pd.read_csv(RESULTS_DIR / 'gradcam_status.csv') if (RESULTS_DIR / 'gradcam_status.csv').exists() else pd.DataFrame()
        gradcam_ok = bool(len(gradcam_df) and gradcam_df.iloc[0].get('status') == 'pass')

        audit_df.to_csv(REPORT_DIR / 'publication_artifact_audit.csv', index=False)
        collapse_df.to_csv(REPORT_DIR / 'prediction_coverage_audit.csv', index=False)
        protocol_audit_df.to_csv(REPORT_DIR / 'protocol_validity_audit.csv', index=False)

        missing_artifacts = audit_df[~audit_df['exists']]['artifact'].tolist()
        collapsed_outputs = collapse_df[collapse_df['status'] == 'COLLAPSED']['file'].tolist() if len(collapse_df) else []
        failed_protocols = protocol_audit_df[~(protocol_audit_df['no_errors'] & protocol_audit_df['metric_columns_present'] & protocol_audit_df['row_count_ok'])]['protocol'].tolist()
        warnings = []
        if not gradcam_ok:
            warnings.append('gradcam_not_available_for_explainability_claims')
        if not tflite_ok:
            failed_protocols.append('tflite_validation')

        verdict = 'PUBLICATION_DATA_READY' if not missing_artifacts and not collapsed_outputs and not failed_protocols else 'PUBLICATION_DATA_HAS_CONCERNS'
        summary = {
            'verdict': verdict,
            'missing_artifacts': missing_artifacts,
            'collapsed_outputs': collapsed_outputs,
            'failed_protocols': failed_protocols,
            'warnings': warnings,
        }
        (REPORT_DIR / 'publication_readiness_summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return summary, audit_df, collapse_df, protocol_audit_df


    readiness_summary, artifact_audit_df, prediction_coverage_audit_df, protocol_validity_audit_df = audit_publication_package()
    display(artifact_audit_df)
    display(prediction_coverage_audit_df.head())
    display(protocol_validity_audit_df)
else:
    print('Skipped (RUN_FULL_ORIGINAL_PIPELINE=False): publication-package readiness audit')


# In[ ]:


if RUN_FULL_ORIGINAL_PIPELINE:
    def iter_package_files():
        selected_paths = [PROJECT_DIR / 'runtime_info.json', PROJECT_DIR / 'dataset_profile.json', PROJECT_DIR / 'final_model_selection.json', RESULTS_DIR, SPLIT_DIR, FIGURE_DIR, REPORT_DIR]
        model_files = [MODEL_DIR / 'final_model.tflite']
        if not KAGGLE_COMPACT_ARTIFACTS:
            selected_paths.extend([MODEL_DIR, MLFLOW_DIR])
        else:
            selected_paths.extend(model_files)
        for selected_path in selected_paths:
            if not selected_path.exists():
                print('Missing, skipped:', selected_path)
                continue
            if selected_path.is_file():
                yield selected_path
                continue
            for path in selected_path.rglob('*'):
                if not path.is_file():
                    continue
                if KAGGLE_COMPACT_ARTIFACTS and path.suffix in {'.keras', '.h5', '.ckpt', '.pb'}:
                    continue
                yield path


    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    zip_path = Path('/kaggle/working') / f'stitchnet_q1_ready_v5_compact_evidence_{timestamp}.zip'
    package_files = list(dict.fromkeys(iter_package_files()))
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in package_files:
            zf.write(path, path.relative_to(PROJECT_DIR))
    print('ZIP CREATED:', zip_path)
    print('Files packaged:', len(package_files))
    print('Size MB:', round(zip_path.stat().st_size / (1024 * 1024), 2))
    try:
        from IPython.display import FileLink, display
        display(FileLink(str(zip_path)))
    except Exception as exc:
        print('FileLink unavailable:', exc)
else:
    print('Skipped (RUN_FULL_ORIGINAL_PIPELINE=False): evidence-package zip packaging')


# ## Expected publication outputs
# 
# The final zip should contain the complete V5 evidence package. Readiness now fails if LODO or cold-start tables contain error rows, missing metric columns, or incomplete row counts.
# 
# Key files:
# 
# - `runtime_info.json`
# - `dataset_profile.json`
# - `splits/p0_6_2_2_by_fabric_thread_seed_0.csv`
# - `splits/p1_lodo_domain_*_seed_*.csv`
# - `splits/p2_cold_start_domain_*_support_*_seed_*.csv`
# - `results/wide_screening_results.csv`
# - `results/systematic_fine_tuning_results.csv`
# - `results/domain_generalisation_results.csv`
# - `results/cold_start_adaptation_results.csv`
# - `results/cold_start_support_summary.csv`
# - `results/robustness_results.csv`
# - `results/calibration_results.csv`
# - `results/risk_coverage_results.csv`
# - `results/edge_device_results.csv`
# - `results/tflite_validation_results.csv`
# - `results/failure_cases.csv`
# - `results/gradcam_status.csv`
# - `models/final_model.tflite`
# - `figures/failure_case_gradcam_gallery.png` only when Grad-CAM succeeds
# - `reports/protocol_validity_audit.csv`
# - `reports/publication_readiness_summary.json`
# 
# If the readiness verdict is `PUBLICATION_DATA_HAS_CONCERNS`, inspect `reports/protocol_validity_audit.csv` and `reports/publication_readiness_summary.json` before writing the manuscript.
# 

# ## Round-3 R&R Addendum — Step 1: Matched-Budget LODO (Comments #1, #4, #7)
# 
# Added for the round-3 major-revision response. Does not modify any cell above —
# the original zero-effort LODO result (`domain_generalisation_results.csv`,
# `DOMAIN_EPOCHS=8`) is preserved exactly as submitted and remains reported in
# the manuscript as the zero-effort ceiling.
# 
# This cell reruns LODO for the best (E), median (G), worst (I), and
# weakest-cold-start-recovery (C) domains at the same epoch budget and learning
# rate as the in-distribution baseline (`HEAD_EPOCHS=30`, `FINETUNE_EPOCHS=40`,
# `FINETUNE_LR=1e-5`), addressing the matched-budget comment directly.
# 
# It also turns on `save_predictions` (for LODO-condition calibration, comment
# #4) and, for domains I and C only, `save_model` (for domain-shift Grad-CAM,
# comment #7). Both parameters already existed on `train_stable_sc` but were set
# to `False` for the original LODO sweep to save Kaggle disk during the 33-run run.
# 

# In[ ]:


# ============================================================
# STEP 1 (R&R comments #1, #4, #7): matched-budget LODO for
# best/median/worst/weakest-recovering domains. Does NOT touch
# domain_generalisation_results.csv -- the original zero-effort
# LODO result stays exactly as submitted.
# ============================================================

MATCHED_BUDGET_DOMAINS = ['E', 'G', 'I', 'C']   # best, median, worst, weakest-recovering
MATCHED_BUDGET_HEAD_EPOCHS = HEAD_EPOCHS        # 30 -- same as in-distribution baseline
MATCHED_BUDGET_FT_EPOCHS = FINETUNE_EPOCHS      # 40 -- same as in-distribution baseline
MATCHED_BUDGET_FT_LR = FINETUNE_LR              # 1e-5 -- same as in-distribution baseline
SAVE_MODEL_DOMAINS = {'I', 'C'}                 # only these need checkpoints, for Grad-CAM

# Resume-skip safety net (added alongside the cleanup-order fix above): a
# fresh execution of this cell used to call out_path.unlink() unconditionally
# and redo all 12 (domain, seed) combos from scratch, discarding the 11
# already-completed runs -- ~80h of GPU time -- to make up the 1 missing
# (C, seed 2) combo. Mirrors the RESUME_FROM_CSV pattern already used by
# Step 2 below.
MATCHED_BUDGET_RESUME_FROM_CSV = RESULTS_DIR / 'domain_generalisation_matched_budget_results.csv'


def run_matched_budget_lodo(model_name):
    rows = []
    completed = set()
    out_path = RESULTS_DIR / 'domain_generalisation_matched_budget_results.csv'
    if MATCHED_BUDGET_RESUME_FROM_CSV is not None and Path(MATCHED_BUDGET_RESUME_FROM_CSV).exists():
        prior = pd.read_csv(MATCHED_BUDGET_RESUME_FROM_CSV)
        prior_ok = prior[prior['error'].isna()] if 'error' in prior.columns else prior
        completed = set(zip(prior_ok['held_out_domain'].astype(str), prior_ok['seed'].astype(int)))
        rows = prior.to_dict('records')
        print(f'RESUME MODE: {len(completed)} matched-budget LODO runs already completed, skipping those.')
    elif out_path.exists():
        out_path.unlink()
    for seed in DEPLOYMENT_PROTOCOL_SEEDS:
        for domain in MATCHED_BUDGET_DOMAINS:
            if (str(domain), int(seed)) in completed:
                continue
            # Recovery path: train_stable_sc already wrote this run's individual
            # save_eval_outputs() files (found for C/seed2 after an interrupted
            # run -- the aggregate CSV lost only the final write_rows() call, not
            # the underlying training). Recover the row instead of re-running an
            # ~8-10h training job for data that already exists on disk.
            run_name = f'{model_name}-S.C.lodo_matched.{domain}.seed{seed}'
            individual_metrics_path = RESULTS_DIR / f'domain_matched_{run_name}_metrics.csv'
            if individual_metrics_path.exists():
                recovered = pd.read_csv(individual_metrics_path).iloc[0].to_dict()
                recovered.update({
                    'held_out_domain': domain, 'seed': seed,
                    'protocol_id': 'p1_leave_one_fabric_thread_domain_out_matched_budget',
                    'epoch_budget': 'matched_to_in_distribution', 'error': np.nan,
                })
                rows.append(recovered)
                write_rows(out_path, rows)
                completed.add((str(domain), int(seed)))
                print(f'RECOVERED (no retrain): {domain}/seed{seed} from {individual_metrics_path.name}')
                continue
            release_keras_memory()
            fold_df = make_lodo_split(df, domain, seed)
            fold_path = SPLIT_DIR / f'p1_lodo_matched_domain_{domain}_seed_{seed}.csv'
            fold_df.to_csv(fold_path, index=False)
            tr = fold_df[fold_df['split'] == 'train'].copy()
            va = fold_df[fold_df['split'] == 'val'].copy()
            te = fold_df[fold_df['split'] == 'test'].copy()
            tr_gen, va_gen, te_gen = make_generators(tr, va, te, batch_size=FOLD_BATCH_SIZE, seed=seed)
            try:
                model, metrics, (y_true, y_pred, y_prob) = train_stable_sc(
                    model_name,
                    tr_gen, va_gen, te_gen,
                    run_suffix=f'S.C.lodo_matched.{domain}.seed{seed}',
                    save_prefix='domain_matched',
                    head_epochs=MATCHED_BUDGET_HEAD_EPOCHS,
                    finetune_epochs=MATCHED_BUDGET_FT_EPOCHS,
                    finetune_lr=MATCHED_BUDGET_FT_LR,
                    save_predictions=True,                       # comment #4: LODO calibration
                    save_model=(domain in SAVE_MODEL_DOMAINS),   # comment #7: Grad-CAM
                )
                metrics.update({
                    'held_out_domain': domain, 'seed': seed,
                    'protocol_id': 'p1_leave_one_fabric_thread_domain_out_matched_budget',
                    'epoch_budget': 'matched_to_in_distribution', 'error': np.nan,
                })
                release_keras_memory(model)
            except Exception as exc:
                metrics = {
                    'model': f'{model_name}-S.C.lodo_matched.{domain}.seed{seed}',
                    'held_out_domain': domain, 'seed': seed,
                    'protocol_id': 'p1_leave_one_fabric_thread_domain_out_matched_budget',
                    'error': str(exc),
                }
            rows.append(metrics)
            write_rows(out_path, rows)
    return pd.DataFrame(rows)

import os
if os.environ.get('STITCHNET_SKIP_STEP1_TRAINING') == '1':
    print('STITCHNET_SKIP_STEP1_TRAINING=1: skipping Step 1 training call (validation run).')
    matched_budget_lodo_df = pd.read_csv(RESULTS_DIR / 'domain_generalisation_matched_budget_results.csv')
else:
    matched_budget_lodo_df = run_matched_budget_lodo(DEPLOYMENT_VALIDITY_MODEL_NAME)
display(matched_budget_lodo_df)


# ## Round-3 R&R Addendum — Step 2: Fixed-Query-Set Cold-Start, FULL 11-domain (Comment #2) — **local JupyterLab, single run**
# 
# **Confound in the original cold-start protocol.** `make_cold_start_split` slices
# `support = shuffled[:support_size]`, `test = shuffled[support_size + val:]` — the
# query/test set changes composition and shrinks as `support_size` grows, confounding
# budget with a moving evaluation target. Fixed below by carving a
# `FIXED_QUERY_PER_CLASS=20` slice **first** from the same seeded per-class shuffle, so
# the query set is identical across every support budget for a given (domain, seed).
# Verified locally with a standalone invariance test before use.
# 
# **Scope — full re-run, not a reviewer-sanctioned subset.** Unlike comment #1
# ("at least for the best, median and worst domains"), comment #2 carries no domain
# floor in the reviewer's wording — "re-run the cold-start experiment" reads as the
# full protocol. The original 198-run sweep (11 domains x 6 support sizes x 3 seeds)
# measured **25.3 hours** total training time in the submitted evidence package. That
# was too long for a single Kaggle session, which is why it was split across two
# Kaggle accounts (`step2_fixed_query_coldstart_accountA/` and `_accountB/`). **This
# notebook instead runs all 11 domains in one continuous session**, on the assumption
# that this JupyterLab server is not capped at Kaggle's typical ~12h session length —
# so the 2-account split is not needed here.
# 
# **Dataset access — kagglehub, not a manual download.** A Kaggle-hosted notebook's
# `/kaggle/input` is a read-only mount provided by Kaggle's own infrastructure, which a
# local/JupyterLab server cannot replicate. The closest local equivalent is
# [`kagglehub`](https://github.com/Kaggle/kagglehub): `kagglehub.dataset_download(...)`
# (added to cell 3 above) authenticates the same way the `kaggle` CLI does and
# downloads the dataset to a local cache directory **automatically, on first access
# only** — no manual `kaggle datasets download` + unzip step, and the cache is reused
# on every subsequent run. This still writes the dataset to local disk once (a true
# zero-disk, fetch-per-image-over-the-network mode would be far too slow across
# 900+ fine-tuning epochs here), but requires no manual download step from you.
# 
# Before running this notebook, make sure one of these is in place on the JupyterLab
# server:
# - `~/.kaggle/kaggle.json` (from your Kaggle account -> Settings -> API -> Create New
#   Token), with `chmod 600 ~/.kaggle/kaggle.json`, **or**
# - the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables set before the kernel
#   starts.
# 
# **Resume-skip as a crash safety net.** With no Kaggle session cap, resume-skip in
# `run_cold_start_fixed_query` below is kept as protection against a local crash, kernel
# restart, or power outage during the ~25h run rather than a session-length workaround:
# if the run is interrupted, point `RESUME_FROM_CSV` at the partially-written
# `cold_start_fixed_query_results.csv` from `RESULTS_DIR` and re-run — already-completed
# (domain, support_size, seed) combinations are skipped. Leave `RESUME_FROM_CSV = None`
# for a fresh start.
# 
# For domains **I** and **C** only (worst LODO domain + the domain that never fully
# recovers under cold-start adaptation — see the ceiling-performance analysis from
# 2026-09-18), model checkpoints are saved at `support_size in {0, 50}` for Step 4
# cold-start Grad-CAM (comment #7 asks for both LODO and cold-start examples).
# 
# Writes to `cold_start_fixed_query_results.csv`. Does not touch the original submitted
# `cold_start_adaptation_results.csv`.

# In[ ]:


# ============================================================
# STEP 2 (R&R comment #2): fixed-query-set cold-start, FULL 11-domain
# scope, single run on this JupyterLab server. Does NOT touch
# cold_start_adaptation_results.csv -- the original submitted cold-start
# result stays exactly as-is.
# ============================================================

COLD_START_FIXED_QUERY_DOMAINS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']   # all 11 domains, single run
FIXED_QUERY_PER_CLASS = 20                      # held constant across all support budgets
SAVE_MODEL_DOMAINS_COLDSTART = {'I', 'C'}       # for Step 4 cold-start Grad-CAM
SAVE_MODEL_SUPPORT_SIZES = {0, 50}              # zero-shot and fully-adapted checkpoints only

# Resume-skip safety net: if this run gets interrupted (crash/kernel restart/power
# loss), point this at the partial cold_start_fixed_query_results.csv written so
# far (it lives at RESULTS_DIR / 'cold_start_fixed_query_results.csv') and re-run.
# e.g. RESUME_FROM_CSV = RESULTS_DIR / 'cold_start_fixed_query_results.csv'
RESUME_FROM_CSV = RESULTS_DIR / 'cold_start_fixed_query_results.csv'


def make_cold_start_split_fixed_query(source_df, held_out_domain, support_size, seed,
                                       adaptation_val_per_class, fixed_query_per_class):
    target = source_df[source_df['fabric_thread'].astype(str) == str(held_out_domain)].copy()
    source = source_df[source_df['fabric_thread'].astype(str) != str(held_out_domain)].copy()
    source['split'] = 'source_train'
    rng = np.random.default_rng(seed)
    parts = [source]
    for _, group in target.groupby('class_id', sort=True):
        shuffled = group.sample(frac=1, random_state=int(rng.integers(0, 2**31 - 1))).copy()
        needed = fixed_query_per_class + support_size + adaptation_val_per_class
        if len(shuffled) < needed:
            return None
        # Query slice is fixed FIRST from the seeded shuffle -> identical across every
        # support_size for this (domain, seed), since the shuffle itself does not
        # depend on support_size.
        query = shuffled.iloc[:fixed_query_per_class].copy()
        remainder = shuffled.iloc[fixed_query_per_class:]
        support = remainder.iloc[:support_size].copy()
        adap_val = remainder.iloc[support_size:support_size + adaptation_val_per_class].copy()
        query['split'] = 'target_final_test'
        support['split'] = 'target_support'
        adap_val['split'] = 'target_adaptation_val'
        parts.extend([query, support, adap_val])
    out = pd.concat(parts, ignore_index=True)
    out['protocol_id'] = 'p2_cold_start_fixed_query'
    out['held_out_domain'] = held_out_domain
    out['support_per_class'] = support_size
    out['adaptation_val_per_class'] = adaptation_val_per_class
    out['fixed_query_per_class'] = fixed_query_per_class
    out['seed'] = seed
    return out


def run_cold_start_fixed_query(model_name):
    import sys
    from datetime import datetime, timedelta

    rows = []
    completed = set()
    out_path = RESULTS_DIR / 'cold_start_fixed_query_results.csv'
    if RESUME_FROM_CSV is not None and Path(RESUME_FROM_CSV).exists():
        prior = pd.read_csv(RESUME_FROM_CSV)
        if 'error' in prior.columns:
            prior_ok = prior[prior['error'].isna()]
        else:
            prior_ok = prior
        completed = set(zip(
            prior_ok['held_out_domain'].astype(str),
            prior_ok['support_per_class'].astype(int),
            prior_ok['seed'].astype(int),
        ))
        rows = prior.to_dict('records')
        print(f'\n{'='*80}')
        print(f'RESUME MODE: {len(completed)} runs already completed, skipping those.')
        print(f'{'='*80}\n')
    if out_path.exists() and RESUME_FROM_CSV is None:
        out_path.unlink()

    # Calculate total runs for monitoring
    total_possible = len(DEPLOYMENT_PROTOCOL_SEEDS) * len(COLD_START_FIXED_QUERY_DOMAINS) * len(SUPPORT_SIZES)
    total_todo = total_possible - len(completed)
    run_counter = [0]
    start_time = time.time()

    print(f'\n{'='*80}')
    print(f'STEP 2: Cold-Start Fixed Query (11 domains × {len(SUPPORT_SIZES)} support sizes × {len(DEPLOYMENT_PROTOCOL_SEEDS)} seeds)')
    print(f'Total: {total_possible} runs | Already done: {len(completed)} | TODO: {total_todo}')
    print(f'Estimated time: ~{total_todo * 0.065:.1f}h (assuming ~3.9min per run)')
    print(f'Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{'='*80}\n')

    for seed in DEPLOYMENT_PROTOCOL_SEEDS:
        for domain in COLD_START_FIXED_QUERY_DOMAINS:
            for support_size in SUPPORT_SIZES:
                if (str(domain), int(support_size), int(seed)) in completed:
                    continue
                release_keras_memory()
                fold_df = make_cold_start_split_fixed_query(
                    df, domain, support_size, seed, ADAPTATION_VAL_PER_CLASS, FIXED_QUERY_PER_CLASS
                )
                if fold_df is None:
                    metrics = {
                        'model': f'{model_name}-S.C.cold_fixedq.{domain}.{support_size}.seed{seed}',
                        'held_out_domain': domain, 'support_per_class': support_size, 'seed': seed,
                        'protocol_id': 'p2_cold_start_fixed_query', 'error': 'insufficient_target_samples',
                    }
                    rows.append(metrics)
                    write_rows(out_path, rows)
                    run_counter[0] += 1
                    elapsed = time.time() - start_time
                    eta_sec = (elapsed / run_counter[0]) * (total_todo - run_counter[0])
                    print(f'[{run_counter[0]:3d}/{total_todo}] SKIP {domain}/support{support_size}/seed{seed} (insufficient_target_samples) | ETA: {timedelta(seconds=int(eta_sec))}')
                    continue
                fold_path = SPLIT_DIR / f'p2_cold_start_fixedq_domain_{domain}_support_{support_size}_seed_{seed}.csv'
                fold_df.to_csv(fold_path, index=False)
                source_train = fold_df[fold_df['split'] == 'source_train'].copy()
                support = fold_df[fold_df['split'] == 'target_support'].copy()
                adap_val = fold_df[fold_df['split'] == 'target_adaptation_val'].copy()
                final_test = fold_df[fold_df['split'] == 'target_final_test'].copy()
                train_part = pd.concat([source_train, support], ignore_index=True) if support_size > 0 else source_train
                tr_gen, va_gen, te_gen = make_generators(train_part, adap_val, final_test, batch_size=FOLD_BATCH_SIZE, seed=seed)
                save_model_flag = (domain in SAVE_MODEL_DOMAINS_COLDSTART) and (support_size in SAVE_MODEL_SUPPORT_SIZES)
                try:
                    model, metrics, _ = train_stable_sc(
                        model_name,
                        tr_gen, va_gen, te_gen,
                        run_suffix=f'S.C.cold_fixedq.{domain}.{support_size}.seed{seed}',
                        save_prefix='cold_fixedq',
                        head_epochs=ADAPT_EPOCHS,
                        finetune_epochs=ADAPT_EPOCHS,
                        finetune_lr=ADAPT_LR,
                        save_predictions=True,
                        save_model=save_model_flag,
                    )
                    metrics.update({
                        'held_out_domain': domain, 'support_per_class': support_size, 'seed': seed,
                        'protocol_id': 'p2_cold_start_fixed_query',
                        'fixed_query_per_class': FIXED_QUERY_PER_CLASS, 'error': np.nan,
                    })
                    release_keras_memory(model)
                    run_counter[0] += 1
                    elapsed = time.time() - start_time
                    eta_sec = (elapsed / run_counter[0]) * (total_todo - run_counter[0])
                    f1 = metrics.get('f1_macro', np.nan)
                    print(f'[{run_counter[0]:3d}/{total_todo}] ✓ {domain}/support{support_size:2d}/seed{seed} | F1={f1:.3f} | ETA: {timedelta(seconds=int(eta_sec))}')
                except Exception as exc:
                    metrics = {
                        'model': f'{model_name}-S.C.cold_fixedq.{domain}.{support_size}.seed{seed}',
                        'held_out_domain': domain, 'support_per_class': support_size, 'seed': seed,
                        'protocol_id': 'p2_cold_start_fixed_query',
                        'fixed_query_per_class': FIXED_QUERY_PER_CLASS, 'error': str(exc),
                    }
                    run_counter[0] += 1
                    elapsed = time.time() - start_time
                    eta_sec = (elapsed / run_counter[0]) * (total_todo - run_counter[0])
                    print(f'[{run_counter[0]:3d}/{total_todo}] ✗ {domain}/support{support_size:2d}/seed{seed} | ERROR | ETA: {timedelta(seconds=int(eta_sec))}')
                rows.append(metrics)
                write_rows(out_path, rows)

    # Final summary
    elapsed = time.time() - start_time
    df_result = pd.DataFrame(rows)
    success_count = len(df_result[df_result['error'].isna()]) if 'error' in df_result.columns else len(df_result)
    error_count = len(df_result) - success_count

    print(f'\n{'='*80}')
    print(f'STEP 2 COMPLETED')
    print(f'Total runs: {len(df_result)} | Success: {success_count} | Errors: {error_count}')
    print(f'Time elapsed: {timedelta(seconds=int(elapsed))}')
    if success_count > 0 and 'f1_macro' in df_result.columns:
        mean_f1 = df_result[df_result['error'].isna()]['f1_macro'].mean()
        print(f'Mean F1 (macro): {mean_f1:.3f}')
    print(f'Results saved to: {out_path}')
    print(f'End: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{'='*80}\n')

    return df_result


cold_start_fixed_query_df = run_cold_start_fixed_query(DEPLOYMENT_VALIDITY_MODEL_NAME)
if 'f1_macro' in cold_start_fixed_query_df:
    cold_start_fixed_query_summary_df = cold_start_fixed_query_df.groupby(['held_out_domain', 'support_per_class'], dropna=False).agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        f1_macro_mean=('f1_macro', 'mean'),
        f1_macro_std=('f1_macro', 'std'),
        n=('model', 'count'),
    ).reset_index()
    cold_start_fixed_query_summary_df.to_csv(RESULTS_DIR / 'cold_start_fixed_query_summary.csv', index=False)
    display(cold_start_fixed_query_summary_df)
display(cold_start_fixed_query_df)



# ## Round-3 R&R Addendum — Step 3: LODO-Condition Calibration (Comment #4)
#
# Computes ECE, NLL (log loss), multiclass Brier score, and the high-confidence
# error rate (fraction of predictions with max softmax probability >=
# `CALIBRATION_HIGH_CONF_THRESHOLD` that are still wrong) from the
# `save_predictions=True` outputs of the Step 1 matched-budget LODO runs
# above. Needs only the saved `domain_matched_*_y_true.npy` /
# `_y_prob.npy` arrays -- no model checkpoint required, so this runs
# regardless of how many of the 12 (domain, seed) combos are done.
#
# Writes `results/calibration_lodo_matched_results.csv` (one row per
# completed run) and `results/calibration_lodo_matched_summary.csv`
# (mean/std across seeds per domain).

# In[ ]:


import re

CALIBRATION_N_BINS = 15
CALIBRATION_HIGH_CONF_THRESHOLD = 0.90
NUM_CLASSES = len(label_order)


def expected_calibration_error(y_true, y_prob, n_bins=CALIBRATION_N_BINS):
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def multiclass_brier_score(y_true, y_prob, n_classes=NUM_CLASSES):
    onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


def high_confidence_error_rate(y_true, y_prob, threshold=CALIBRATION_HIGH_CONF_THRESHOLD):
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    mask = confidences >= threshold
    n_high_conf = int(mask.sum())
    if n_high_conf == 0:
        return n_high_conf, np.nan
    return n_high_conf, float((predictions[mask] != y_true[mask]).mean())


def compute_calibration_row(prefix, run_name, held_out_domain, seed, protocol_id):
    y_true_path = RESULTS_DIR / f'{prefix}_{run_name}_y_true.npy'
    y_prob_path = RESULTS_DIR / f'{prefix}_{run_name}_y_prob.npy'
    if not (y_true_path.exists() and y_prob_path.exists()):
        return None
    y_true = np.load(y_true_path)
    y_prob = np.load(y_prob_path)
    n_high_conf, high_conf_err = high_confidence_error_rate(y_true, y_prob)
    return {
        'held_out_domain': held_out_domain,
        'seed': seed,
        'protocol_id': protocol_id,
        'n': int(len(y_true)),
        'ece': expected_calibration_error(y_true, y_prob),
        'nll': float(log_loss(y_true, y_prob, labels=list(range(NUM_CLASSES)))),
        'brier': multiclass_brier_score(y_true, y_prob),
        'high_conf_threshold': CALIBRATION_HIGH_CONF_THRESHOLD,
        'high_conf_n': n_high_conf,
        'high_conf_error_rate': high_conf_err,
    }


def run_lodo_matched_calibration():
    rows = []
    pattern = re.compile(r'^domain_matched_(?P<run_name>.+\.lodo_matched\.(?P<domain>[A-Za-z0-9]+)\.seed(?P<seed>\d+))_y_true\.npy$')
    for y_true_path in sorted(RESULTS_DIR.glob('domain_matched_*_y_true.npy')):
        match = pattern.match(y_true_path.name)
        if not match:
            continue
        row = compute_calibration_row(
            'domain_matched', match.group('run_name'),
            match.group('domain'), int(match.group('seed')),
            'p1_leave_one_fabric_thread_domain_out_matched_budget',
        )
        if row is not None:
            rows.append(row)
    result_df = pd.DataFrame(rows).sort_values(['held_out_domain', 'seed']).reset_index(drop=True)
    result_df.to_csv(RESULTS_DIR / 'calibration_lodo_matched_results.csv', index=False)
    if len(result_df):
        summary_df = result_df.groupby('held_out_domain').agg(
            ece_mean=('ece', 'mean'), ece_std=('ece', 'std'),
            nll_mean=('nll', 'mean'), nll_std=('nll', 'std'),
            brier_mean=('brier', 'mean'), brier_std=('brier', 'std'),
            high_conf_error_rate_mean=('high_conf_error_rate', 'mean'),
            high_conf_error_rate_std=('high_conf_error_rate', 'std'),
            n_runs=('seed', 'count'),
        ).reset_index()
        summary_df.to_csv(RESULTS_DIR / 'calibration_lodo_matched_summary.csv', index=False)
        display(summary_df)
    expected_combos = {(d, s) for d in MATCHED_BUDGET_DOMAINS for s in DEPLOYMENT_PROTOCOL_SEEDS}
    found_combos = set(zip(result_df['held_out_domain'], result_df['seed'])) if len(result_df) else set()
    missing = sorted(expected_combos - found_combos)
    print(f'LODO-matched calibration: {len(found_combos)}/{len(expected_combos)} combos covered.')
    if missing:
        print(f'Missing (domain, seed) combos (no saved predictions yet): {missing}')
    return result_df


lodo_matched_calibration_df = run_lodo_matched_calibration()
display(lodo_matched_calibration_df)


# ## Round-3 R&R Addendum — Step 3b: Checkpoint-Only Recovery for LODO Grad-CAM (Comment #7)
#
# The `domain_matched_*.keras` checkpoints for I/C were destroyed by the
# cleanup-ordering bug fixed above (see Step 1). Retraining full-scope
# (2 domains x 3 seeds = 6 runs, matched-budget epochs -> ~48-60h GPU) was
# judged not worth it for what is fundamentally a qualitative, illustrative
# Grad-CAM figure. Scope agreed: **domain I and C, seed 0 only** (2 runs,
# ~16-20h), giving one LODO domain-shift example per domain.
#
# Writes to a **separate** `domain_matched_gradcam_ckpt_*` prefix rather than
# `domain_matched_*` -- this is a fresh training run of an already-reported
# (domain, seed) combo, so it must not silently overwrite the canonical
# `domain_matched_*` metrics/predictions that `calibration_lodo_matched_results.csv`
# above was already computed from (retraining is not perfectly deterministic
# even with a fixed seed, given cuDNN op nondeterminism). Step 4 below checks
# this prefix as a fallback only when the canonical LODO checkpoint is absent.

# In[ ]:


LODO_GRADCAM_CHECKPOINT_ONLY = [('I', 0), ('C', 0)]
LODO_GRADCAM_ONLY_PREFIX = 'domain_matched_gradcam_ckpt'


def run_lodo_gradcam_checkpoint_only(model_name):
    status_rows = []
    for domain, seed in LODO_GRADCAM_CHECKPOINT_ONLY:
        run_name = f'{model_name}-S.C.lodo_matched.{domain}.seed{seed}'
        model_path = MODEL_DIR / f'{LODO_GRADCAM_ONLY_PREFIX}_{run_name}.keras'
        if model_path.exists():
            print(f'SKIP (checkpoint already recovered): {domain}/seed{seed}')
            status_rows.append({'held_out_domain': domain, 'seed': seed, 'status': 'already_present'})
            continue
        release_keras_memory()
        fold_df = make_lodo_split(df, domain, seed)
        tr = fold_df[fold_df['split'] == 'train'].copy()
        va = fold_df[fold_df['split'] == 'val'].copy()
        te = fold_df[fold_df['split'] == 'test'].copy()
        tr_gen, va_gen, te_gen = make_generators(tr, va, te, batch_size=FOLD_BATCH_SIZE, seed=seed)
        try:
            model, metrics, _ = train_stable_sc(
                model_name, tr_gen, va_gen, te_gen,
                run_suffix=f'S.C.lodo_matched.{domain}.seed{seed}',
                save_prefix=LODO_GRADCAM_ONLY_PREFIX,
                head_epochs=MATCHED_BUDGET_HEAD_EPOCHS,
                finetune_epochs=MATCHED_BUDGET_FT_EPOCHS,
                finetune_lr=MATCHED_BUDGET_FT_LR,
                save_predictions=True,
                save_model=True,
            )
            release_keras_memory(model)
            status_rows.append({'held_out_domain': domain, 'seed': seed, 'status': 'trained', 'f1_macro': metrics.get('f1_macro')})
            print(f'Checkpoint recovered: {domain}/seed{seed} | f1_macro={metrics.get("f1_macro")}')
        except Exception as exc:
            status_rows.append({'held_out_domain': domain, 'seed': seed, 'status': f'failed:{exc}'})
    return pd.DataFrame(status_rows)


lodo_gradcam_checkpoint_status_df = run_lodo_gradcam_checkpoint_only(DEPLOYMENT_VALIDITY_MODEL_NAME)
display(lodo_gradcam_checkpoint_status_df)


# ## Round-3 R&R Addendum — Step 4: Grad-CAM for Poorly-Transferring LODO and Cold-Start Runs (Comment #7)
#
# Generates Grad-CAM overlays from whichever `domain_matched_*.keras` (LODO,
# comment #7's "domain-shift" half) and `cold_fixedq_*.keras` (cold-start
# half) checkpoints currently exist in `models/`. Each checkpoint's held-out
# target split is rebuilt deterministically (same seed, same split function
# used at training time) purely to recover file paths in the same row order
# as the already-saved `_y_true.npy`/`_y_pred.npy` arrays; the checkpoint
# itself supplies the actual forward/backward pass Grad-CAM needs.
#
# For each checkpoint, up to `GRADCAM_MAX_EXAMPLES` of the most-confidently
# **wrong** predictions are rendered (falling back to the lowest-confidence
# correct predictions if the run made no errors) -- these are the
# "poorly-transferring" examples the comment asks for. Writes one gallery PNG
# per checkpoint to `figures/` and a `results/gradcam_status.csv` audit table
# (comment #7 also requires: failed/skipped Grad-CAM is recorded explicitly,
# never silently treated as evidence).

# In[ ]:


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from PIL import Image as PILImage

GRADCAM_LAST_CONV_LAYER = {'MobileNetV1': 'conv_pw_13_relu'}


def load_image_for_gradcam(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    return arr / 255.0, arr.astype('uint8')


def make_gradcam_heatmap(img_batch, model, last_conv_layer_name, pred_index):
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_batch)
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(image_uint8, heatmap, alpha=0.4):
    heatmap_img = PILImage.fromarray((heatmap * 255).astype('uint8')).resize((image_uint8.shape[1], image_uint8.shape[0]))
    heatmap_color = (mpl_cm.jet(np.array(heatmap_img) / 255.0)[..., :3] * 255).astype('uint8')
    return (image_uint8 * (1 - alpha) + heatmap_color * alpha).astype('uint8')


def rebuild_target_filepaths(kind, domain, seed, support_size=None):
    """Reproduce the exact held-out target split used at training time, deterministically, to recover filepaths in save order."""
    if kind == 'lodo':
        fold_df = make_lodo_split(df, domain, seed)
        target = fold_df[fold_df['split'] == 'test'].copy()
    else:
        fold_df = make_cold_start_split_fixed_query(df, domain, support_size, seed, ADAPTATION_VAL_PER_CLASS, FIXED_QUERY_PER_CLASS)
        if fold_df is None:
            return None
        target = fold_df[fold_df['split'] == 'target_final_test'].copy()
    return target.reset_index(drop=True)


def run_gradcam_for_checkpoint(model_path, kind, domain, seed, support_size, prefix, run_name):
    status_row = {
        'model_path': str(model_path), 'kind': kind, 'held_out_domain': domain,
        'seed': seed, 'support_per_class': support_size, 'status': None, 'n_examples': 0, 'figure_path': None,
    }
    y_true_path = RESULTS_DIR / f'{prefix}_{run_name}_y_true.npy'
    y_prob_path = RESULTS_DIR / f'{prefix}_{run_name}_y_prob.npy'
    if not model_path.exists():
        status_row['status'] = 'skipped_missing_checkpoint'
        return status_row
    if not (y_true_path.exists() and y_prob_path.exists()):
        status_row['status'] = 'skipped_missing_predictions'
        return status_row
    target_df = rebuild_target_filepaths(kind, domain, seed, support_size)
    if target_df is None:
        status_row['status'] = 'skipped_insufficient_target_samples'
        return status_row
    y_true = np.load(y_true_path)
    y_prob = np.load(y_prob_path)
    if len(target_df) != len(y_true):
        status_row['status'] = f'skipped_row_count_mismatch(target_df={len(target_df)},y_true={len(y_true)})'
        return status_row
    y_pred = y_prob.argmax(axis=1)
    confidence = y_prob.max(axis=1)
    wrong_idx = np.where(y_pred != y_true)[0]
    if len(wrong_idx):
        order = wrong_idx[np.argsort(-confidence[wrong_idx])]
        example_label = 'misclassified'
    else:
        order = np.argsort(confidence)
        example_label = 'lowest_confidence_correct'
    example_idx = order[:GRADCAM_MAX_EXAMPLES]

    try:
        model = tf.keras.models.load_model(model_path)
        last_conv_layer_name = GRADCAM_LAST_CONV_LAYER.get(DEPLOYMENT_VALIDITY_MODEL_NAME)
        fig, axes = plt.subplots(1, len(example_idx), figsize=(4 * len(example_idx), 4.5))
        axes = np.atleast_1d(axes)
        for ax, idx in zip(axes, example_idx):
            filepath = target_df.iloc[int(idx)]['filepath']
            img_float, img_uint8 = load_image_for_gradcam(filepath)
            heatmap = make_gradcam_heatmap(img_float[np.newaxis, ...], model, last_conv_layer_name, int(y_pred[idx]))
            overlay = overlay_gradcam(img_uint8, heatmap)
            ax.imshow(overlay)
            ax.set_title(f'true={class_names[int(y_true[idx])]}\npred={class_names[int(y_pred[idx])]} ({confidence[idx]:.2f})', fontsize=9)
            ax.axis('off')
        fig.suptitle(f'{kind.upper()} | domain={domain} seed={seed}' + (f' support={support_size}' if support_size is not None else '') + f' | {example_label}')
        fig.tight_layout()
        fig_path = FIGURE_DIR / f'gradcam_{prefix}_{run_name}.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        release_keras_memory(model)
        status_row.update({'status': 'generated', 'n_examples': int(len(example_idx)), 'figure_path': str(fig_path)})
    except Exception as exc:
        status_row['status'] = f'failed:{exc}'
    return status_row


def run_all_gradcam():
    status_rows = []
    for domain in sorted(SAVE_MODEL_DOMAINS):
        for seed in DEPLOYMENT_PROTOCOL_SEEDS:
            run_name = f'S.C.lodo_matched.{domain}.seed{seed}'
            full_run_name = f'{DEPLOYMENT_VALIDITY_MODEL_NAME}-{run_name}'
            canonical_path = MODEL_DIR / f'domain_matched_{full_run_name}.keras'
            # Canonical checkpoint (from the originally-reported training run) is
            # preferred; fall back to the checkpoint-only recovery run (Step 3b)
            # only when the canonical one is missing (comment #7 checkpoints
            # wiped by the cleanup-order bug -- see Step 1).
            if canonical_path.exists():
                model_path, prefix = canonical_path, 'domain_matched'
            else:
                model_path, prefix = MODEL_DIR / f'{LODO_GRADCAM_ONLY_PREFIX}_{full_run_name}.keras', LODO_GRADCAM_ONLY_PREFIX
            status_rows.append(run_gradcam_for_checkpoint(model_path, 'lodo', domain, seed, None, prefix, full_run_name))
    for domain in sorted(SAVE_MODEL_DOMAINS_COLDSTART):
        for support_size in sorted(SAVE_MODEL_SUPPORT_SIZES):
            for seed in DEPLOYMENT_PROTOCOL_SEEDS:
                run_name = f'S.C.cold_fixedq.{domain}.{support_size}.seed{seed}'
                model_path = MODEL_DIR / f'cold_fixedq_{DEPLOYMENT_VALIDITY_MODEL_NAME}-{run_name}.keras'
                status_rows.append(run_gradcam_for_checkpoint(model_path, 'cold_start', domain, seed, support_size, 'cold_fixedq', f'{DEPLOYMENT_VALIDITY_MODEL_NAME}-{run_name}'))
    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(RESULTS_DIR / 'gradcam_status.csv', index=False)
    print(status_df['status'].value_counts(dropna=False).to_string())
    return status_df


gradcam_status_df = run_all_gradcam()
display(gradcam_status_df)
