# -*- coding: utf-8 -*-

"""
Evaluate the newly trained VQA model on test set
"""

import torch
import os
from dataset_RAD import VQAFeatureDataset
from tools.create_dictionary import Dictionary
from torch.utils.data import DataLoader
from multi_level_model import BAN_Model
from classify_question import classify_model
import argparse
from train import compute_score_with_logits
import logging

# Setup logger
log_dir = 'log_eval'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join(log_dir, 'eval_vqa_model.log'))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VQA Model")
    parser.add_argument('--gpu', type=int, default=-1, help='GPU device ID (-1 for CPU)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--model_path', type=str, default='saved_models/2025Oct28-005928/checkpoint_epoch_99.pth', 
                        help='path to trained VQA model')
    parser.add_argument('--question_classifier_path', type=str, default='saved_models/type_classifier.pth',
                        help='path to question classifier')
    
    # Model args
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--hid_dim', type=int, default=1024)
    parser.add_argument('--v_dim', type=int, default=64)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--attention', type=str, default='BAN')
    parser.add_argument('--glimpse', type=int, default=2)
    parser.add_argument('--use_counter', type=bool, default=False)
    parser.add_argument('--num_stacks', type=int, default=2)
    parser.add_argument('--rnn', type=str, default='GRU')
    parser.add_argument('--question_len', type=int, default=12)
    parser.add_argument('--tfidf', type=bool, default=True)
    parser.add_argument('--cat', type=bool, default=True)
    parser.add_argument('--eps_cnn', type=float, default=1e-5)
    parser.add_argument('--momentum_cnn', type=float, default=0.05)
    parser.add_argument('--maml', type=bool, default=True)
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights')
    parser.add_argument('--autoencoder', type=bool, default=True)
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth')
    parser.add_argument('--ae_alpha', type=float, default=0.001)
    parser.add_argument('--other_model', type=bool, default=False)
    parser.add_argument('--use_data', type=bool, default=True)
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')
    
    return args

def evaluate_vqa_model(args):
    logger.info("="*60)
    logger.info("VQA Model Evaluation on Test Set")
    logger.info("="*60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Classifier path: {args.question_classifier_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Load dictionary
    print("Loading dictionary...")
    dictionary = Dictionary.load_from_file(os.path.join(args.data_dir, 'dictionary.pkl'))
    
    # Load dataset
    print("Loading test dataset...")
    test_set = VQAFeatureDataset('test', args, dictionary, dataroot=args.data_dir)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logger.info(f"Test samples: {len(test_set)}")
    
    # Load VQA model
    print("Loading VQA model...")
    model = BAN_Model(test_set, args)
    model = model.to(args.device)
    
    # Load trained weights
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found at {args.model_path}")
        return
    
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # Handle both formats: dict with 'model_state' and direct state_dict
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        logger.info(f"Model loaded from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Model loaded successfully")
    
    # Evaluation
    logger.info("\nEvaluating on test set...")
    model.eval()
    
    total_score = 0
    total_samples = 0
    
    closed_score = 0
    closed_total = 0
    
    open_score = 0
    open_total = 0
    
    with torch.no_grad():
        for batch_idx, (v, q, a, answer_type, question_type, phrase_type, answer_target) in enumerate(test_loader):
            # Prepare data
            if args.maml:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(args.device)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(args.device)
            
            q = q.to(args.device)
            a = a.to(args.device)
            
            # Forward pass
            if args.autoencoder:
                last_output_close, last_output_open, a_close, a_open, decoder = model(v, q, a, answer_target)
            else:
                last_output_close, last_output_open, a_close, a_open = model(v, q, a, answer_target)
            
            # Get predictions
            preds_close, preds_open = model.classify(last_output_close, last_output_open)
            
            # Compute scores
            batch_close_score = 0.0
            batch_open_score = 0.0
            
            if preds_close.shape[0] > 0:
                batch_close_score = compute_score_with_logits(preds_close, a_close.data).sum().item()
                closed_score += batch_close_score
                closed_total += preds_close.shape[0]
            
            if preds_open.shape[0] > 0:
                batch_open_score = compute_score_with_logits(preds_open, a_open.data).sum().item()
                open_score += batch_open_score
                open_total += preds_open.shape[0]
            
            total_score += batch_close_score + batch_open_score
            total_samples += q.shape[0]
    
    # Calculate accuracies
    total_accuracy = 100 * total_score / total_samples if total_samples > 0 else 0
    closed_accuracy = 100 * closed_score / closed_total if closed_total > 0 else 0
    open_accuracy = 100 * open_score / open_total if open_total > 0 else 0
    
    logger.info("="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Total Test Samples: {total_samples}")
    logger.info(f"Total Accuracy: {total_accuracy:.2f}%")
    logger.info("")
    logger.info(f"CLOSED-ended Questions:")
    logger.info(f"  Samples: {closed_total}")
    logger.info(f"  Correct: {int(closed_score)}")
    logger.info(f"  Accuracy: {closed_accuracy:.2f}%")
    logger.info("")
    logger.info(f"OPEN-ended Questions:")
    logger.info(f"  Samples: {open_total}")
    logger.info(f"  Correct: {int(open_score)}")
    logger.info(f"  Accuracy: {open_accuracy:.2f}%")
    logger.info("="*60)
    
    print("\n" + "="*60)
    print(f"Total Test Accuracy: {total_accuracy:.2f}%")
    print(f"CLOSED Accuracy: {closed_accuracy:.2f}% ({int(closed_score)}/{closed_total})")
    print(f"OPEN Accuracy: {open_accuracy:.2f}% ({int(open_score)}/{open_total})")
    print("="*60)

if __name__ == '__main__':
    args = parse_args()
    print(f"Using device: {args.device}")
    evaluate_vqa_model(args)
