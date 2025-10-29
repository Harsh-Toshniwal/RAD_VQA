# -*- coding: utf-8 -*-
"""
Train model with validation set and then predict on test data
"""
import os
import sys
import json
import torch
import argparse
import numpy as np
from collections import defaultdict

from main import create_model_and_loaders
from train import train, evaluate_classifier
from utils import create_dir
import utils


def save_trained_model(model, optimizer, epoch, output_dir, model_name='trained_model.pth'):
    """Save the trained model and optimizer state"""
    create_dir(output_dir)
    model_path = os.path.join(output_dir, model_name)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")
    return model_path


def load_trained_model(model, model_path, device):
    """Load the trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    print(f"Model loaded from {model_path} (epoch {epoch})")
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
            
            # Process batch
            batch_size = q.shape[0]
            for i in range(batch_size):
                # Get question info
                q_id = batch_idx * args.batch_size + i
                
                # Get ground truth
                gt_type = question_type[i]
                
                # Determine predicted answer
                if gt_type == 'CLOSED':
                    pred_logits = preds_close[sum(question_type[:i] == 'CLOSED')].unsqueeze(0)
                    pred_idx = torch.argmax(pred_logits, dim=1).item()
                    predicted_answer = dataset.ans_candidates[pred_idx]
                    confidence = torch.sigmoid(pred_logits[0, pred_idx]).item()
                else:
                    pred_logits = preds_open[sum(question_type[:i] == 'OPEN')].unsqueeze(0)
                    pred_idx = torch.argmax(pred_logits, dim=1).item()
                    if pred_idx < len(dataset.ans_candidates):
                        predicted_answer = dataset.ans_candidates[pred_idx]
                    else:
                        predicted_answer = "unknown"
                    confidence = torch.sigmoid(pred_logits[0, pred_idx]).item()
                
                # Get ground truth answer
                gt_answer = dataset.ans_candidates[a[i].item()] if a[i].item() < len(dataset.ans_candidates) else "unknown"
                
                # Check if correct
                is_correct = predicted_answer == gt_answer
                if is_correct:
                    total_correct += 1
                total_samples += 1
                
                # Store prediction
                pred_dict = {
                    'question_id': q_id,
                    'question': dataset.questions[q_id] if q_id < len(dataset.questions) else "unknown",
                    'predicted_answer': predicted_answer,
                    'ground_truth': gt_answer,
                    'question_type_pred': gt_type,
                    'question_type_gt': gt_type,
                    'correct': is_correct,
                    'confidence': confidence
                }
                predictions.append(pred_dict)
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return predictions, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use (-1 for CPU)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--output', type=str, default='saved_models', help='Output directory')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu < 0:
        args.device = torch.device('cpu')
        print("Using CPU")
    else:
        args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {args.device}")
    
    # Create models and loaders
    print("Loading data and models...")
    model, question_classify, train_loader, val_loader, test_loader, dataset = create_model_and_loaders(args)
    
    # Train the model with validation
    print(f"Training model for {args.epochs} epochs...")
    optim = torch.optim.Adamax(params=model.parameters(), lr=args.lr)
    train(args, model, question_classify, train_loader, val_loader)
    
    # Save the trained model
    print("\nSaving trained model...")
    model_save_dir = os.path.join(args.output, 'trained_weights')
    create_dir(model_save_dir)
    model_path = save_trained_model(model, optim, args.epochs - 1, model_save_dir)
    
    # Load the trained model (verify it loads correctly)
    print("\nLoading trained model for predictions...")
    model_loaded = load_trained_model(model, model_path, args.device)
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    predictions, accuracy = predict_on_test_set(model_loaded, question_classify, test_loader, args, dataset)
    
    # Save predictions
    output_file = 'predictions_output_trained.json'
    result = {
        'accuracy': accuracy,
        'total_samples': len(predictions),
        'correct': sum(1 for p in predictions if p['correct']),
        'predictions': predictions
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nPredictions saved to {output_file}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Total samples: {len(predictions)}")
    print(f"Correct predictions: {sum(1 for p in predictions if p['correct'])}")


if __name__ == '__main__':
    main()
