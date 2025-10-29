#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for the trained VQA model on test set
Compares with baseline predictions to show improvement
"""

import os
import sys
import torch
import json
import argparse
import logging
import _pickle as cPickle
from datetime import datetime
from pathlib import Path

# Import local modules
from dataset_RAD import VQAFeatureDataset
from multi_level_model import BAN_Model
from classifier import SimpleClassifier
from main import parse_args
import utils

def setup_logger(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'eval_trained_model.log')
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def compute_score_with_logits(logits, labels):
    """Compute accuracy score"""
    import torch
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size(), dtype=labels.dtype, device=labels.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def evaluate_model(model, test_loader, device, logger):
    """Evaluate model on test set"""
    model.eval()
    
    total_score = 0.0
    total_samples = 0
    open_score = 0.0
    open_count = 0
    close_score = 0.0
    close_count = 0
    
    with torch.no_grad():
        for i, (v, q, a, answer_type, question_type, phrase_type, answer_target) in enumerate(test_loader):
            # Prepare images
            if v[0] is not None:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if v[1] is not None:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            
            q = q.to(device)
            a = a.to(device)
            
            # Forward pass
            last_output_close, last_output_open, a_close, a_open, decoder = model(v, q, a, answer_target)
            preds_close, preds_open = model.classify(last_output_close, last_output_open)
            
            # Compute scores
            batch_close_score = 0.0
            batch_open_score = 0.0
            
            if preds_close.shape[0] > 0:
                batch_close_score = compute_score_with_logits(preds_close, a_close.data).sum()
                close_score += batch_close_score
                close_count += preds_close.shape[0]
            
            if preds_open.shape[0] > 0:
                batch_open_score = compute_score_with_logits(preds_open, a_open.data).sum()
                open_score += batch_open_score
                open_count += preds_open.shape[0]
            
            total_score += batch_close_score + batch_open_score
            total_samples += q.shape[0]
            
            if (i + 1) % 5 == 0:
                logger.info(f"Evaluated {i+1} batches ({total_samples} samples)...")
    
    # Calculate final metrics
    overall_acc = 100 * total_score / total_samples if total_samples > 0 else 0
    open_acc = 100 * open_score / open_count if open_count > 0 else 0
    close_acc = 100 * close_score / close_count if close_count > 0 else 0
    
    return {
        'overall_acc': overall_acc,
        'open_acc': open_acc,
        'close_acc': close_acc,
        'total_samples': total_samples,
        'open_samples': open_count,
        'close_samples': close_count
    }

def main():
    # Parse arguments
    args = parse_args()
    args.gpu = -1  # Force CPU
    device = torch.device('cpu')
    args.device = device
    
    # Setup logger
    logger = setup_logger('log_eval')
    
    logger.info("=" * 60)
    logger.info("Evaluating Trained VQA Model")
    logger.info("=" * 60)
    
    # Load dictionary
    logger.info("Loading dictionary...")
    data_dir = args.data_dir
    d = cPickle.load(open(os.path.join(data_dir, 'dictionary.pkl'), 'rb'))
    
    # Load dataset
    logger.info("Loading test dataset...")
    dataset = VQAFeatureDataset('test', args, d, dataroot=data_dir)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    logger.info(f"Test set size: {len(dataset)} samples")
    
    # Load model
    logger.info("Loading model architecture...")
    model = BAN_Model(args, dataset)
    model = model.to(device)
    
    # Find and load the latest checkpoint
    saved_models_dir = 'saved_models'
    checkpoint_loaded = False
    
    if os.path.exists(saved_models_dir):
        # Get all directories sorted by modification time
        model_dirs = sorted(
            [d for d in os.listdir(saved_models_dir) if os.path.isdir(os.path.join(saved_models_dir, d))],
            key=lambda x: os.path.getmtime(os.path.join(saved_models_dir, x)),
            reverse=True
        )
        
        if model_dirs:
            latest_model_dir = os.path.join(saved_models_dir, model_dirs[0])
            logger.info(f"Using latest model directory: {latest_model_dir}")
            
            # Find checkpoint files
            pth_files = [f for f in os.listdir(latest_model_dir) if f.endswith('.pth')]
            if pth_files:
                ckpt_path = os.path.join(latest_model_dir, pth_files[0])
                logger.info(f"Loading checkpoint: {ckpt_path}")
                
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Checkpoint from epoch {checkpoint['epoch_id']}")
                checkpoint_loaded = True
            else:
                logger.warning("No checkpoint found, using untrained model")
        else:
            logger.warning("No saved models found, using untrained model")
    else:
        logger.warning("saved_models directory not found, using untrained model")
    
    # Evaluate
    logger.info("=" * 60)
    logger.info("Starting evaluation on test set...")
    logger.info("=" * 60)
    
    results = evaluate_model(model, test_loader, device, logger)
    
    # Log results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {results['overall_acc']:.2f}%")
    logger.info(f"  - Open-ended:   {results['open_acc']:.2f}% ({results['open_samples']} samples)")
    logger.info(f"  - Close-ended:  {results['close_acc']:.2f}% ({results['close_samples']} samples)")
    logger.info(f"Total test samples: {results['total_samples']}")
    logger.info("=" * 60)
    
    # Comparison with baseline
    logger.info("\nComparison with Baseline (Rule-based):")
    logger.info(f"  Baseline Overall: 31.49%")
    logger.info(f"  Trained Overall:  {results['overall_acc']:.2f}%")
    logger.info(f"  Improvement:      +{results['overall_acc'] - 31.49:.2f}%")
    logger.info(f"\n  Baseline Close:   52.40%")
    logger.info(f"  Trained Close:    {results['close_acc']:.2f}%")
    logger.info(f"  Improvement:      +{results['close_acc'] - 52.40:.2f}%")
    
    return results

if __name__ == '__main__':
    results = main()
