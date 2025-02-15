#!/usr/bin/env python3

import pandas as pd
import wandb
from sklearn.metrics import classification_report
from pprint import pprint


def evaluate(model, file_path):
    
    df = pd.read_csv(file_path)

    model_input = df[['src', 'mt']].to_dict('records')
    
    df['prediction'] = model.predict(model_input, batch_size=8, gpus=1)['scores']
    
    metrics = {
        f'{file_path}_spearman': df['score'].corr(df['prediction'], method='spearman'),
        f'{file_path}_kendall': df['score'].corr(df['prediction'], method='kendall'),
        f'{file_path}_pearson': df['score'].corr(df['prediction'], method='pearson'),
    }
    
    if 'critical'in df.columns and 'major' in df.columns:
        df['is_bad'] = (df['critical'] > 0) | (df['major'] > 0)
        df['is_bad_pred'] = df['prediction'] < df['prediction'].quantile(0.2)
        
        report = classification_report(df['is_bad'], df['is_bad_pred'], output_dict=True)
        
        metrics[f'{file_path}_recall@bottom20'] = report['True']['recall']
        metrics[f'{file_path}_report'] = report
    
    if wandb.run is not None:
        wandb.log(metrics)
    else:
        pprint(metrics)
    
    return metrics

if __name__ == '__main__':
    import argparse
    from comet import load_from_checkpoint

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--file_path', type=str, required=True)
    args = parser.parse_args()

    model = load_from_checkpoint(args.model_path)
    evaluate(model, args.file_path)