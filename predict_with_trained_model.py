# -*- coding: utf-8 -*-
"""
Make VQA predictions on test data using the trained model
"""
import os
import sys
import torch
import json
import pickle
import argparse
import numpy as np
from torch.utils.data import DataLoader
from multi_level_model import BAN_Model
from classifier import SimpleClassifier
from dataset_RAD import Dictionary, VQAFeatureDataset
from train import evaluate

def main():
    parser = argparse.ArgumentParser(description='VQA Predictions with Trained Model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id (-1 for CPU)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output_file', type=str, default='trained_model_predictions.json',
                        help='Output file for predictions')
    args = parser.parse_args()

    # Set device
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}')
    
    print(f"Device: {device}")
    print(f"Loading data from: {args.data_dir}")
    
    # Load dictionary
    print("Loading dictionary...")
    d = Dictionary.load_from_file(os.path.join(args.data_dir, 'dictionary.pkl'))
    
    # Create args for dataset
    class DatasetArgs:
        def __init__(self):
            self.maml = True
            self.autoencoder = True
            self.other_model = False
            self.v_dim = 64
            self.hid_dim = 1024
            self.ae_model_path = os.path.join(args.data_dir, 'pretrained_ae.pth')
            self.ae_alpha = 0.001
            self.maml_model_path = os.path.join(args.data_dir, 'pretrained_maml.weights')
    
    dataset_args = DatasetArgs()
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = VQAFeatureDataset(
        'test',
        dataset_args,
        d,
        dataroot=args.data_dir
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Num close answer candidates: {test_dataset.num_close_candidates}")
    print(f"Num open answer candidates: {test_dataset.num_open_candidates}")
    
    # Create model with same args as training
    class ModelArgs:
        def __init__(self):
            self.dropout = 0.5
            self.attention = 'BAN'
            self.glimpse = 2
            self.use_counter = False
            self.num_stacks = 2
            self.rnn = 'GRU'
            self.question_len = 12
            self.tfidf = True
            self.cat = True
            self.hid_dim = 1024
            self.v_dim = 64
            self.autoencoder = True
            self.ae_model_path = 'pretrained_ae.pth'
            self.ae_alpha = 0.001
            self.maml = True
            self.maml_model_path = 'pretrained_maml.weights'
            self.other_model = False
            self.activation = 'relu'
            self.device = device
            self.data_dir = args.data_dir
            self.eps_cnn = 1e-5
            self.momentum_cnn = 0.05
            self.use_data = True
    
    model_args = ModelArgs()
    
    print("Creating BAN model...")
    model = BAN_Model(
        test_dataset,
        model_args
    )
    model = model.to(device)
    
    # Load question classifier
    print("Loading question classifier...")
    question_classify = SimpleClassifier(
        d.ntoken,
        model_args.hid_dim,
        2,
        model_args
    )
    question_classify = question_classify.to(device)
    
    # Load pretrained question classifier
    ckpt_path = os.path.join(args.data_dir, 'type_classifier.pth')
    if os.path.exists(ckpt_path):
        print(f"Loading question classifier from {ckpt_path}...")
        pretrained_model = torch.load(ckpt_path, map_location=device)
        question_classify.load_state_dict(pretrained_model)
    
    print("\n" + "="*60)
    print("Evaluating on test data")
    print("="*60)
    
    model.eval()
    question_classify.eval()
    
    predictions = []
    all_scores = []
    
    with torch.no_grad():
        for batch_idx, (v, q, a, answer_type, question_type, phrase_type, answer_target) in enumerate(test_loader):
            # Move to device and reshape
            if model_args.maml:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if model_args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            if model_args.other_model:
                v = v.to(device)
            
            q = q.to(device)
            a = a.to(device)
            
            # Forward pass
            if model_args.autoencoder:
                last_output_close, last_output_open, a_close, a_open, decoder = model(v, q, a, answer_target)
            else:
                last_output_close, last_output_open, a_close, a_open = model(v, q, a, answer_target)
            
            preds_close, preds_open = model.classify(last_output_close, last_output_open)
            
            # Collect predictions from this batch
            # preds_close and preds_open only contain predictions for their respective types
            close_idx = 0
            open_idx = 0
            
            for i in range(q.shape[0]):
                ans_type = answer_type[i] if isinstance(answer_type[i], str) else answer_type[i].item()
                is_closed = ans_type == 'CLOSED' or ans_type == 0
                
                if is_closed and close_idx < preds_close.shape[0]:
                    pred_logits = preds_close[close_idx]
                    pred_idx = torch.argmax(pred_logits).item()
                    pred_answer = test_dataset.label2close[pred_idx]
                    confidence = torch.softmax(pred_logits.unsqueeze(0), dim=1)[0, pred_idx].item()
                    close_idx += 1
                elif not is_closed and open_idx < preds_open.shape[0]:
                    pred_logits = preds_open[open_idx]
                    pred_idx = torch.argmax(pred_logits).item()
                    pred_answer = test_dataset.label2open[pred_idx]
                    confidence = torch.softmax(pred_logits.unsqueeze(0), dim=1)[0, pred_idx].item()
                    open_idx += 1
                else:
                    pred_answer = 'UNKNOWN'
                    confidence = 0.0
                
                predictions.append({
                    'answer_type': 'CLOSED' if is_closed else 'OPEN',
                    'prediction': pred_answer,
                    'confidence': float(confidence),
                    'ground_truth': 'UNKNOWN'
                })
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {(batch_idx + 1) * args.batch_size}/{len(test_dataset)} samples")
    
    # Save predictions
    print(f"\nSaving predictions to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Print statistics
    closed_correct = sum(1 for p in predictions if p['answer_type'] == 'CLOSED' and p['prediction'] == p['ground_truth'])
    open_correct = sum(1 for p in predictions if p['answer_type'] == 'OPEN' and p['prediction'] == p['ground_truth'])
    closed_total = sum(1 for p in predictions if p['answer_type'] == 'CLOSED')
    open_total = sum(1 for p in predictions if p['answer_type'] == 'OPEN')
    
    print("\n" + "="*60)
    print("Prediction Results")
    print("="*60)
    print(f"Total predictions: {len(predictions)}")
    print(f"CLOSED questions: {closed_total} ({closed_correct} correct = {100*closed_correct/closed_total:.2f}%)")
    print(f"OPEN questions: {open_total} ({open_correct} correct = {100*open_correct/open_total:.2f}%)")
    print(f"Overall accuracy: {100*(closed_correct + open_correct)/len(predictions):.2f}%")
    print(f"\nSample predictions:")
    for i in range(min(5, len(predictions))):
        p = predictions[i]
        match = "✓" if p['prediction'] == p['ground_truth'] else "✗"
        print(f"  [{i+1}] {match} Type:{p['answer_type']:6s} | Pred:{p['prediction']:20s} | GT:{p['ground_truth']:20s} | Conf:{p['confidence']:.3f}")

if __name__ == '__main__':
    main()
