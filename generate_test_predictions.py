# -*- coding: utf-8 -*-

"""
Generate detailed test predictions with ground truth comparison
"""

import torch
import os
import json
from dataset_RAD import VQAFeatureDataset
from tools.create_dictionary import Dictionary
from torch.utils.data import DataLoader
from multi_level_model import BAN_Model
import argparse
from train import compute_score_with_logits
import logging

# Setup logger
log_dir = 'log_eval'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join(log_dir, 'predictions.log'))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Test Predictions")
    parser.add_argument('--gpu', type=int, default=-1, help='GPU device ID (-1 for CPU)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--model_path', type=str, default='saved_models/2025Oct28-005928/checkpoint_epoch_99.pth', 
                        help='path to trained VQA model')
    
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

def generate_predictions(args):
    logger.info("="*60)
    logger.info("Generating Test Set Predictions")
    logger.info("="*60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Device: {args.device}")
    
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
    
    # Generate predictions
    logger.info("\nGenerating predictions on test set...")
    model.eval()
    
    predictions = []
    sample_idx = 0
    
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
            
            # Process each sample in batch
            for i in range(q.shape[0]):
                # Determine question type
                if i < preds_close.shape[0]:
                    # CLOSED question
                    pred_logits = preds_close[i].detach().cpu()
                    pred_idx = torch.argmax(pred_logits).item()
                    pred_conf = torch.softmax(pred_logits, dim=0)[pred_idx].item()
                    
                    # Ground truth
                    gt_idx = torch.argmax(a_close[i]).item()
                    is_correct = (pred_idx == gt_idx)
                    
                    # Get answer text
                    pred_answer = test_set.label2ans[pred_idx]
                    gt_answer = test_set.label2ans[gt_idx]
                    q_type = "CLOSED"
                    
                elif i < preds_open.shape[0]:
                    # OPEN question
                    pred_logits = preds_open[i].detach().cpu()
                    pred_idx = torch.argmax(pred_logits).item()
                    pred_conf = torch.softmax(pred_logits, dim=0)[pred_idx].item()
                    
                    # Ground truth
                    gt_idx = torch.argmax(a_open[i]).item()
                    is_correct = (pred_idx == gt_idx)
                    
                    # Get answer text
                    pred_answer = test_set.label2ans[pred_idx + 56]  # Offset for open answers
                    gt_answer = test_set.label2ans[gt_idx + 56]
                    q_type = "OPEN"
                else:
                    continue
                
                # Get question text
                question_tokens = q[i].detach().cpu().numpy().tolist()
                question_text = ' '.join([dictionary.idx2word[idx] if idx < len(dictionary.idx2word) else '<UNK>' 
                                         for idx in question_tokens if idx < len(dictionary.idx2word)])
                
                prediction_record = {
                    "sample_idx": sample_idx,
                    "question": question_text.strip(),
                    "question_type": q_type,
                    "answer_type": answer_type[i] if hasattr(answer_type, '__getitem__') else str(answer_type),
                    "predicted_answer": pred_answer,
                    "predicted_idx": int(pred_idx),
                    "predicted_confidence": float(pred_conf),
                    "ground_truth_answer": gt_answer,
                    "ground_truth_idx": int(gt_idx),
                    "is_correct": bool(is_correct),
                    "batch_idx": batch_idx,
                    "position_in_batch": i
                }
                
                predictions.append(prediction_record)
                sample_idx += 1
    
    logger.info(f"Generated {len(predictions)} predictions")
    
    # Calculate statistics
    correct_count = sum(1 for p in predictions if p['is_correct'])
    total_count = len(predictions)
    overall_acc = correct_count / total_count if total_count > 0 else 0
    
    closed_preds = [p for p in predictions if p['question_type'] == 'CLOSED']
    open_preds = [p for p in predictions if p['question_type'] == 'OPEN']
    
    closed_acc = sum(1 for p in closed_preds if p['is_correct']) / len(closed_preds) if closed_preds else 0
    open_acc = sum(1 for p in open_preds if p['is_correct']) / len(open_preds) if open_preds else 0
    
    # Create summary
    summary = {
        "model_info": {
            "model_path": args.model_path,
            "device": str(args.device),
            "batch_size": args.batch_size
        },
        "statistics": {
            "total_samples": total_count,
            "correct_predictions": correct_count,
            "overall_accuracy": float(overall_acc),
            "closed_count": len(closed_preds),
            "closed_correct": sum(1 for p in closed_preds if p['is_correct']),
            "closed_accuracy": float(closed_acc),
            "open_count": len(open_preds),
            "open_correct": sum(1 for p in open_preds if p['is_correct']),
            "open_accuracy": float(open_acc)
        },
        "predictions": predictions
    }
    
    # Save to JSON
    output_file = 'test_predictions.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Predictions saved to {output_file}")
    logger.info("="*60)
    logger.info("PREDICTION STATISTICS")
    logger.info("="*60)
    logger.info(f"Total Samples: {total_count}")
    logger.info(f"Overall Accuracy: {overall_acc*100:.2f}%")
    logger.info(f"CLOSED Questions: {closed_acc*100:.2f}% ({sum(1 for p in closed_preds if p['is_correct'])}/{len(closed_preds)})")
    logger.info(f"OPEN Questions: {open_acc*100:.2f}% ({sum(1 for p in open_preds if p['is_correct'])}/{len(open_preds)})")
    logger.info("="*60)
    
    print("\n" + "="*60)
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    print(f"CLOSED Accuracy: {closed_acc*100:.2f}%")
    print(f"OPEN Accuracy: {open_acc*100:.2f}%")
    print(f"Predictions saved to: {output_file}")
    print("="*60)

if __name__ == '__main__':
    args = parse_args()
    print(f"Using device: {args.device}")
    generate_predictions(args)
