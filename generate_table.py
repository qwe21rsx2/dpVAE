import argparse
import csv
import json
import os
import os.path as osp

import numpy as np
import torch

from trainers.trainer import get_trainer


DEFAULT_CONFIG = 'config/beta/mimic_sepsis.json'
DEFAULT_OUTPUT = 'outputs/mimic_sepsis/generated_10000.csv'


def load_table_metadata(config):
    feature_cols = config.get('feature_cols')
    categorical_cols = set(config.get('categorical_cols', []))
    exclude_cols = set(config.get('exclude_cols', []))

    with open(config['dset_dir'], newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames

    if feature_cols is None:
        feature_cols = [col for col in columns if col not in exclude_cols]

    encoders = {}
    data = []
    for row in rows:
        values = []
        for col in feature_cols:
            value = row[col]
            if value == '':
                values.append(np.nan)
            elif col in categorical_cols:
                mapping = encoders.setdefault(col, {})
                if value not in mapping:
                    mapping[value] = len(mapping)
                values.append(float(mapping[value]))
            else:
                values.append(float(value))
        data.append(values)

    data = np.asarray(data, dtype=np.float32)
    means = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(means, inds[1])

    rng = np.random.RandomState(config.get('seed', 1))
    indices = np.arange(len(data))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - config.get('val_ratio', 0.2)))
    train_idx = indices[:split]

    if config.get('normalize', True):
        mean = data[train_idx].mean(axis=0)
        std = data[train_idx].std(axis=0)
        std[std == 0] = 1
    else:
        mean = np.zeros(data.shape[1], dtype=np.float32)
        std = np.ones(data.shape[1], dtype=np.float32)

    decoders = {
        col: {idx: value for value, idx in mapping.items()}
        for col, mapping in encoders.items()
    }

    return {
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'decoders': decoders,
        'mean': mean.astype(np.float32),
        'std': std.astype(np.float32),
    }


def decode_generated(samples, metadata):
    samples = samples * metadata['std'] + metadata['mean']
    rows = []

    for values in samples:
        row = {}
        for idx, col in enumerate(metadata['feature_cols']):
            value = values[idx]
            if col in metadata['categorical_cols']:
                mapping = metadata['decoders'][col]
                cat_idx = int(np.rint(value))
                cat_idx = min(max(cat_idx, 0), max(mapping.keys()))
                row[col] = mapping[cat_idx]
            else:
                row[col] = float(value)
        rows.append(row)

    return rows


def generate(config_path, checkpoint, num_samples, output_path, batch_size):
    with open(config_path) as f:
        config = json.load(f)

    checkpoint_path = osp.join(config['ckpt_dir'], checkpoint)
    if not osp.isfile(checkpoint_path):
        raise FileNotFoundError('Checkpoint not found: {}'.format(checkpoint_path))

    config['cont'] = True
    config['ckpt_name'] = checkpoint

    trainer = get_trainer(config)
    metadata = load_table_metadata(config)

    generated = []
    remaining = num_samples
    with torch.no_grad():
        while remaining > 0:
            current = min(batch_size, remaining)
            batch = trainer.generate_sample(current)
            generated.append(batch.detach().cpu().numpy())
            remaining -= current

    generated = np.concatenate(generated, axis=0)
    rows = decode_generated(generated, metadata)

    os.makedirs(osp.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metadata['feature_cols'])
        writer.writeheader()
        writer.writerows(rows)

    np.save(osp.splitext(output_path)[0] + '_normalized.npy', generated)
    print('Generated {} rows: {}'.format(num_samples, output_path))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic table rows from a trained VAE checkpoint.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', default='best')
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--batch-size', type=int, default=1000)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate(args.config, args.checkpoint, args.num_samples, args.output, args.batch_size)
