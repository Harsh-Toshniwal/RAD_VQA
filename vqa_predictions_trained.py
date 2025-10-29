# -*- coding: utf-8 -*-
"""
Save a trained model checkpoint and make predictions with it
"""
import os
import torch
import json
import pickle
import argparse
import numpy as np
from torch.utils.data import DataLoader
from multi_level_model import BAN_Model
from classifier import SimpleClassifier
from dataset_RAD import Dictionary, VQAFeatureDataset

def load_model(model, checkpoint_path, device):
    """Load model weights from checkpoint"""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict):
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
            elif 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt)
        print("✓ Checkpoint loaded successfully")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description='Train and Predict VQA with Model Saving')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id (-1 for CPU)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='saved_models/2025Oct27-234345',
                        help='Directory with training checkpoint')
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
            self.ae_model_path = 'pretrained_ae.pth'
            self.ae_alpha = 0.001
            self.maml_model_path = 'pretrained_maml.weights'
    
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
    
    # Create model args
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
    
    # Create model
    print("Creating BAN model...")
    model = BAN_Model(test_dataset, model_args)
    model = model.to(device)
    
    # Try to load checkpoint
    checkpoint_files = [
        os.path.join(args.checkpoint_dir, '9.pth'),  # Best epoch (epoch 9)
        os.path.join(args.checkpoint_dir, '0.pth'),  # Fallback to epoch 0
    ]
    
    checkpoint_loaded = False
    for ckpt_file in checkpoint_files:
        if load_model(model, ckpt_file, device):
            checkpoint_loaded = True
            break
    
    if not checkpoint_loaded:
        print("⚠ Warning: No checkpoint found, using model with pretrained encoders only")
    
    # Load question classifier
    print("Loading question classifier...")
    question_classify = SimpleClassifier(
        d.ntoken,
        model_args.hid_dim,
        2,
        model_args
    )
    question_classify = question_classify.to(device)
    
    ckpt_path = os.path.join(args.data_dir, 'type_classifier.pth')
    if os.path.exists(ckpt_path):
        pretrained_model = torch.load(ckpt_path, map_location=device)
        question_classify.load_state_dict(pretrained_model)
    
    print("\n" + "="*70)
    print("Making Predictions on Test Data")
    print("="*70)
    
    model.eval()
    question_classify.eval()
    
    predictions = []
    
    with torch.no_grad():
        for batch_idx, (v, q, a, answer_type, question_type, phrase_type, answer_target) in enumerate(test_loader):
            # Move to device and reshape
            if model_args.maml:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if model_args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            
            q = q.to(device)
            a = a.to(device)
            
            # Forward pass
            if model_args.autoencoder:
                last_output_close, last_output_open, a_close, a_open, decoder = model(v, q, a, answer_target)
            else:
                last_output_close, last_output_open, a_close, a_open = model(v, q, a, answer_target)
            
            preds_close, preds_open = model.classify(last_output_close, last_output_open)
            
            # Collect predictions - preds_close and preds_open are already filtered by type
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
                })
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {min((batch_idx + 1) * args.batch_size, len(test_dataset))}/{len(test_dataset)} samples")
    
    # Save predictions
    print(f"\nSaving {len(predictions)} predictions to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Print statistics
    closed_preds = [p for p in predictions if p['answer_type'] == 'CLOSED']
    open_preds = [p for p in predictions if p['answer_type'] == 'OPEN']
    
    avg_closed_conf = np.mean([p['confidence'] for p in closed_preds]) if closed_preds else 0
    avg_open_conf = np.mean([p['confidence'] for p in open_preds]) if open_preds else 0
    
    print("\n" + "="*70)
    print("Prediction Summary")
    print("="*70)
    print(f"Total predictions: {len(predictions)}")
    print(f"  CLOSED: {len(closed_preds)} predictions (avg confidence: {avg_closed_conf:.3f})")
    print(f"  OPEN:   {len(open_preds)} predictions (avg confidence: {avg_open_conf:.3f})")
    print(f"\nSample predictions (first 5):")
    for i in range(min(5, len(predictions))):
        p = predictions[i]
        print(f"  [{i+1}] Type:{p['answer_type']:6s} | Answer:{p['prediction']:30s} | Confidence:{p['confidence']:.4f}")
    print("="*70)

if __name__ == '__main__':
    main()
