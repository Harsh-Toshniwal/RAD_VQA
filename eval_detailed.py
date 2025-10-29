# -*- coding: utf-8 -*-
"""
Detailed evaluation script showing predictions vs ground truth
"""

import os
import torch
from torch.utils.data import DataLoader
import utils
from dataset_RAD import Dictionary, VQAFeatureDataset
from classify_question import classify_model
import json


def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description="Med VQA Classifier Detailed Evaluation")
    parser.add_argument('--seed', type=int, default=5, help='random seed. default:5')
    parser.add_argument('--gpu', type=int, default=0, help='use gpu device. default:0')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for evaluation')
    parser.add_argument('--show_samples', type=int, default=20, help='number of samples to show')
    
    # Vision flags (needed by dataset)
    parser.add_argument('--v_dim', default=64, type=int, help='visual feature dim')
    parser.add_argument('--autoencoder', action='store_true', default=True)
    parser.add_argument('--maml', action='store_true', default=True)
    parser.add_argument('--other_model', action='store_true', default=False)
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights')
    
    args = parser.parse_args()
    return args


def evaluate_detailed(model, dataloader, dictionary, device, show_samples=20):
    """
    Evaluate model and show detailed predictions
    
    Returns:
        score: accuracy percentage
        predictions: list of (question_text, ground_truth, prediction, correct)
    """
    score = 0
    number = 0
    predictions = []
    
    # Label mapping for question types
    label_to_type = {0: 'OPEN', 1: 'CLOSED'}
    
    model.eval()
    with torch.no_grad():
        for i, row in enumerate(dataloader):
            image_data, question, target, answer_type, question_type, phrase_type, answer_target = row
            question_tokens = question.clone()  # Save for decoding
            question, answer_target = question.to(device), answer_target.to(device)
            
            # Forward pass
            output = model(question)
            
            # Get predictions
            pred = output.data.max(1)[1]
            
            # Calculate accuracy
            correct = pred.eq(answer_target.data).cpu().sum()
            score += correct.item()
            number += len(answer_target)
            
            # Store detailed predictions
            for j in range(len(answer_target)):
                gt_label = answer_target[j].item()
                pred_label = pred[j].item()
                is_correct = (gt_label == pred_label)
                
                # Decode question tokens to text
                question_ids = question_tokens[j].cpu().numpy()
                question_words = []
                for token_id in question_ids:
                    if token_id != 0:  # Skip padding
                        # idx2word is a list, not dict
                        if token_id < len(dictionary.idx2word):
                            word = dictionary.idx2word[token_id]
                        else:
                            word = '<UNK>'
                        question_words.append(word)
                question_text = ' '.join(question_words)
                
                # Get raw output scores
                output_scores = output[j].cpu().numpy()
                
                predictions.append({
                    'question': question_text,
                    'ground_truth': label_to_type[gt_label],
                    'prediction': label_to_type[pred_label],
                    'correct': is_correct,
                    'output_scores': {
                        'OPEN': float(output_scores[0]),
                        'CLOSED': float(output_scores[1])
                    },
                    'confidence': float(output_scores[pred_label])
                })
    
    accuracy = score / number * 100.
    return accuracy, predictions


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(root, 'data')
    
    args = parse_args()
    args.data_dir = data

    # Set device
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    args.device = device
    
    print(f"Using device: {device}")

    # Fixed random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load dictionary
    dict_path = os.path.join(data, 'dictionary.pkl')
    d = Dictionary.load_from_file(dict_path)
    print(f"Dictionary loaded: {d.ntoken} tokens\n")

    # Build test dataset/loader
    print("Loading test dataset...")
    test_dataset = VQAFeatureDataset('test', args, d, dataroot=data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)
    print(f"Test dataset size: {len(test_dataset)}\n")

    # Build classifier model
    glove_path = os.path.join(data, 'glove6b_init_300d.npy')
    net = classify_model(d.ntoken, glove_path)
    net = net.to(device)

    # Load checkpoint
    ckpt_path = os.path.join(root, 'saved_models', 'type_classifier.pth')
    print(f"Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    
    try:
        net.load_state_dict(state)
    except:
        net.load_state_dict(state.get('model_state', state))
    
    print("Model loaded successfully\n")

    # Evaluate with details
    print("Evaluating on test set...\n")
    print("=" * 80)
    acc, predictions = evaluate_detailed(net, test_loader, d, device, args.show_samples)
    
    print(f"OVERALL ACCURACY: {acc:.2f}% ({int(acc * len(predictions) / 100)}/{len(predictions)} correct)")
    print("=" * 80)
    print("\nMODEL OUTPUT EXPLANATION:")
    print("- The model outputs 2 raw scores (logits) for each class: [OPEN, CLOSED]")
    print("- Higher score = model is more confident about that class")
    print("- Predicted label = class with highest score")
    print("=" * 80)
    
    # Show sample predictions
    print(f"\nSHOWING FIRST {args.show_samples} TEST SAMPLES:")
    print("=" * 80)
    
    for idx, pred_info in enumerate(predictions[:args.show_samples]):
        status = "✓" if pred_info['correct'] else "✗"
        print(f"\nSample {idx + 1} {status}")
        print(f"Question: {pred_info['question']}")
        print(f"Ground Truth: {pred_info['ground_truth']}")
        print(f"Prediction:   {pred_info['prediction']}")
        print(f"Model Output Scores: OPEN={pred_info['output_scores']['OPEN']:.4f}, "
              f"CLOSED={pred_info['output_scores']['CLOSED']:.4f}")
        print(f"Confidence: {pred_info['confidence']:.4f}")
    
    # Show error cases
    errors = [p for p in predictions if not p['correct']]
    if errors:
        print("\n" + "=" * 80)
        print(f"ERROR ANALYSIS: {len(errors)} mistakes found")
        print("=" * 80)
        for idx, error in enumerate(errors[:10]):  # Show first 10 errors
            print(f"\nError {idx + 1}:")
            print(f"Question: {error['question']}")
            print(f"Ground Truth: {error['ground_truth']}")
            print(f"Prediction:   {error['prediction']}")
            print(f"Output Scores: OPEN={error['output_scores']['OPEN']:.4f}, "
                  f"CLOSED={error['output_scores']['CLOSED']:.4f}")
    else:
        print("\n" + "=" * 80)
        print("NO ERRORS! Perfect accuracy!")
        print("=" * 80)
    
    # Save detailed results to JSON
    output_path = os.path.join(root, 'detailed_predictions.json')
    with open(output_path, 'w') as f:
        json.dump({
            'accuracy': acc,
            'total_samples': len(predictions),
            'correct': int(acc * len(predictions) / 100),
            'errors': len(errors),
            'predictions': predictions
        }, f, indent=2)
    
    print(f"\nDetailed predictions saved to: {output_path}")


if __name__ == '__main__':
    main()
