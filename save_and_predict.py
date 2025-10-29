# -*- coding: utf-8 -*-
"""
Save the trained model from the last training run and make predictions on test data
"""
import os
import sys
import json
import torch
import argparse
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from dataset_RAD import VQAFeatureDataset
from tools.create_dictionary import Dictionary
from torch.utils.data import DataLoader
import utils
from multi_level_model import BAN_Model
from classify_question import classify_model


def create_model_and_loaders(args):
    """Create model and data loaders"""
    # Load dictionary
    dictionary = Dictionary.load_from_file(os.path.join(args.data_dir, 'dictionary.pkl'))
    
    # Create dataset
    train_set = VQAFeatureDataset('train', args, dictionary, dataroot=args.data_dir)
    test_set = VQAFeatureDataset('test', args, dictionary, dataroot=args.data_dir)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    # Create model
    model = BAN_Model(train_set, args)
    
    # Load question classifier
    question_classify = classify_model(dictionary.ntoken, os.path.join(args.data_dir, 'glove6b_init_300d.npy'))
    # Try to find type_classifier.pth
    ckpt = os.path.join(args.output, 'type_classifier.pth')
    if not os.path.exists(ckpt):
        ckpt = os.path.join(args.data_dir, 'type_classifier.pth')
    if not os.path.exists(ckpt):
        ckpt = './saved_models/type_classifier.pth'
    pretrained_model = torch.load(ckpt, map_location=args.device)
    question_classify.load_state_dict(pretrained_model)
    
    return model, question_classify, train_loader, test_loader, train_set


def save_trained_model(model, epoch, output_dir, model_name='trained_model.pth'):
    """Save the trained model"""
    utils.create_dir(output_dir)
    model_path = os.path.join(output_dir, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")
    return model_path


def load_trained_model(model, model_path, device):
    """Load the trained model from checkpoint"""
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✓ Model loaded from {model_path}")
    return model


def predict_on_test_set(model, question_model, test_loader, args, dataset):
    """Make predictions on test set"""
    device = args.device
    model.eval()
    question_model.eval()
    
    predictions = []
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (v, q, a, answer_type, question_type, phrase_type, answer_target) in enumerate(test_loader):
            # Prepare data
            if args.maml:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            if args.other_model:
                v = v.to(device)
            
            q = q.to(device)
            a = a.to(device)
            
            # Forward pass
            if args.autoencoder:
                last_output_close, last_output_open, a_close, a_open, decoder = model.forward_classify(
                    v, q, a, question_model
                )
            else:
                last_output_close, last_output_open, a_close, a_open = model.forward_classify(
                    v, q, a, question_model
                )
            
            # Get predictions
            preds_close, preds_open = model.classify(last_output_close, last_output_open)
            
            # Get indices for this batch
            batch_start_idx = batch_idx * args.batch_size
            
            # Track question types and map to answer indices
            close_indices = [j for j, t in enumerate(answer_type) if t == 'CLOSED']
            open_indices = [j for j, t in enumerate(answer_type) if t == 'OPEN']
            
            # Process closed-ended questions
            for close_rel_idx, batch_q_idx in enumerate(close_indices):
                if close_rel_idx >= preds_close.shape[0]:
                    print(f"Warning: close_rel_idx {close_rel_idx} >= preds_close size {preds_close.shape[0]}")
                    continue
                sample_idx = batch_start_idx + batch_q_idx
                if sample_idx >= len(dataset):
                    continue
                
                entry = dataset.entries[sample_idx]
                q_id = entry.get('qid', sample_idx)
                question_text = entry.get('question', 'unknown')
                
                # Get prediction
                pred_logits = preds_close[close_rel_idx]
                pred_idx = torch.argmax(pred_logits).item()
                predicted_answer = dataset.label2close[pred_idx] if pred_idx < len(dataset.label2close) else "unknown"
                
                # Get ground truth
                gt_answer_vec = a_close[close_rel_idx]
                gt_answer_idx = torch.argmax(gt_answer_vec).item()
                gt_answer = dataset.label2close[gt_answer_idx] if gt_answer_idx < len(dataset.label2close) else "unknown"
                
                # Check if correct
                is_correct = predicted_answer == gt_answer
                if is_correct:
                    total_correct += 1
                total_samples += 1
                
                # Store prediction
                pred_dict = {
                    'question_id': int(q_id) if isinstance(q_id, (int, float)) else q_id,
                    'question': question_text,
                    'predicted_answer': predicted_answer,
                    'ground_truth': gt_answer,
                    'question_type_pred': 'CLOSED',
                    'question_type_gt': 'CLOSED',
                    'correct': is_correct
                }
                predictions.append(pred_dict)
            
            # Process open-ended questions
            for open_rel_idx, batch_q_idx in enumerate(open_indices):
                if open_rel_idx >= preds_open.shape[0]:
                    print(f"Warning: open_rel_idx {open_rel_idx} >= preds_open size {preds_open.shape[0]}")
                    continue
                sample_idx = batch_start_idx + batch_q_idx
                if sample_idx >= len(dataset):
                    continue
                
                entry = dataset.entries[sample_idx]
                q_id = entry.get('qid', sample_idx)
                question_text = entry.get('question', 'unknown')
                
                # Get prediction
                pred_logits = preds_open[open_rel_idx]
                pred_idx = torch.argmax(pred_logits).item()
                predicted_answer = dataset.label2open[pred_idx] if pred_idx < len(dataset.label2open) else "unknown"
                
                # Get ground truth
                gt_answer_vec = a_open[open_rel_idx]
                gt_answer_idx = torch.argmax(gt_answer_vec).item()
                gt_answer = dataset.label2open[gt_answer_idx] if gt_answer_idx < len(dataset.label2open) else "unknown"
                
                # Check if correct
                is_correct = predicted_answer == gt_answer
                if is_correct:
                    total_correct += 1
                total_samples += 1
                
                # Store prediction
                pred_dict = {
                    'question_id': int(q_id) if isinstance(q_id, (int, float)) else q_id,
                    'question': question_text,
                    'predicted_answer': predicted_answer,
                    'ground_truth': gt_answer,
                    'question_type_pred': 'OPEN',
                    'question_type_gt': 'OPEN',
                    'correct': is_correct
                }
                predictions.append(pred_dict)
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return predictions, accuracy, total_correct


def main():
    parser = argparse.ArgumentParser(description="Save trained model and make predictions")
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use (-1 for CPU)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output', type=str, default='saved_models', help='Output directory for saving models')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    
    # Model arguments (from main.py)
    parser.add_argument('--seed', type=int, default=5, help='random seed')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--update_freq', default='1', metavar='N')
    parser.add_argument('--print_interval', default=20, type=int, metavar='N')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM')
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn')
    parser.add_argument('--use_data', type=bool, default=True, help='use extra data')
    parser.add_argument('--activation', default='relu', type=str, metavar='activation')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout')
    parser.add_argument('--attention', default='BAN', type=str, metavar='attention')
    parser.add_argument('--glimpse', type=int, default=2)
    parser.add_argument('--use_counter', type=bool, default=False)
    parser.add_argument('--num_stacks', type=int, default=2)
    parser.add_argument('--rnn', default='GRU', type=str, metavar='rnn')
    parser.add_argument('--question_len', type=int, default=12)
    parser.add_argument('--tfidf', type=bool, default=True)
    parser.add_argument('--cat', type=bool, default=True)
    parser.add_argument('--hid_dim', type=int, default=1024)
    parser.add_argument('--v_dim', type=int, default=64)
    parser.add_argument('--autoencoder', type=bool, default=True)
    parser.add_argument('--ae_model_path', default='pretrained_ae.pth')
    parser.add_argument('--ae_alpha', type=float, default=0.001)
    parser.add_argument('--maml', type=bool, default=True)
    parser.add_argument('--maml_model_path', default='pretrained_maml.weights')
    parser.add_argument('--other_model', type=bool, default=False)
    parser.add_argument('--details', default='original ', type=str, metavar='details')
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu < 0:
        args.device = torch.device('cpu')
        print("Using CPU")
    else:
        args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {args.device}\n")
    
    # Create models and loaders
    print("=" * 80)
    print("STEP 1: Loading data and models...")
    print("=" * 80)
    model, question_classify, train_loader, test_loader, dataset = create_model_and_loaders(args)
    model = model.to(args.device)
    question_classify = question_classify.to(args.device)
    print(f"✓ Loaded test set with {len(test_loader.dataset)} samples")
    print(f"✓ Model ready for predictions\n")
    
    # Save the current model (after last training)
    print("=" * 80)
    print("STEP 2: Saving trained model...")
    print("=" * 80)
    model_save_dir = os.path.join(args.output, 'trained_weights')
    model_path = save_trained_model(model, args.epochs - 1, model_save_dir, 'vqa_model_10epochs.pth')
    print()
    
    # Load the trained model (verify it loads correctly)
    print("=" * 80)
    print("STEP 3: Loading trained model for inference...")
    print("=" * 80)
    model = load_trained_model(model, model_path, args.device)
    print()
    
    # Make predictions on test set
    print("=" * 80)
    print("STEP 4: Making predictions on test set...")
    print("=" * 80)
    predictions, accuracy, num_correct = predict_on_test_set(model, question_classify, test_loader, args, dataset)
    print()
    
    # Save predictions
    output_file = 'predictions_trained_model.json'
    result = {
        'accuracy': accuracy,
        'total_samples': len(predictions),
        'correct': num_correct,
        'predictions': predictions
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"✓ Predictions saved to {output_file}")
    print(f"✓ Accuracy: {accuracy * 100:.2f}%")
    print(f"✓ Total samples: {len(predictions)}")
    print(f"✓ Correct predictions: {num_correct}")
    print(f"✓ Model checkpoint: {model_path}")
    print()


if __name__ == '__main__':
    main()
