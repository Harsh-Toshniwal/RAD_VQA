# -*- coding: utf-8 -*-
"""
Make predictions on test images and questions
Answers VQA questions using image features and question text
"""

import os
import torch
import json
from torch.utils.data import DataLoader
import utils
from dataset_RAD import Dictionary, VQAFeatureDataset
from classify_question import classify_model
from multi_level_model import BAN_Model
import argparse


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Med VQA Predictions")
    parser.add_argument('--seed', type=int, default=5, help='random seed. default:5')
    parser.add_argument('--gpu', type=int, default=0, help='use gpu device. default:0')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--save_predictions', type=str, default='predictions_output.json',
                        help='output file for predictions')
    parser.add_argument('--show_samples', type=int, default=20, help='number of samples to show')
    
    # Model loading
    parser.add_argument('--model_path', type=str, default=None,
                        help='path to trained VQA model checkpoint')
    parser.add_argument('--classifier_path', type=str, default='saved_models/type_classifier.pth',
                        help='path to question classifier')
    parser.add_argument('--use_data', action='store_true', default=True,
                         help='Using RAD dataset')
    
    # Vision flags
    parser.add_argument('--v_dim', default=64, type=int, help='visual feature dim')
    parser.add_argument('--autoencoder', action='store_true', default=True)
    parser.add_argument('--maml', action='store_true', default=True)
    parser.add_argument('--other_model', action='store_true', default=False)
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights')
    
    # Model architecture
    parser.add_argument('--attention', type=str, default='BAN')
    parser.add_argument('--glimpse', type=int, default=2)
    parser.add_argument('--use_counter', action='store_true', default=False)
    parser.add_argument('--rnn', type=str, default='GRU')
    parser.add_argument('--question_len', default=12, type=int)
    parser.add_argument('--tfidf', type=bool, default=True)
    parser.add_argument('--cat', type=bool, default=True)
    parser.add_argument('--hid_dim', type=int, default=1024)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dropout', default=0.5, type=float)
    
    # CNN parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float)
    parser.add_argument('--momentum_cnn', default=0.05, type=float)
    parser.add_argument('--ae_alpha', default=0.001, type=float)
    
    args = parser.parse_args()
    return args


def predict(model, question_classifier, dataloader, dataset, dictionary, device, show_samples=20):
    """
    Make predictions on test set
    
    Returns:
        predictions: list of prediction dictionaries
    """
    predictions = []
    model.eval()
    question_classifier.eval()
    
    print("\nMaking predictions on test set...")
    print("=" * 80)
    
    with torch.no_grad():
        for batch_idx, row in enumerate(dataloader):
            image_data, question, target, answer_type, question_type, phrase_type, answer_target = row
            
            # Prepare inputs
            v = image_data
            if args.maml:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1).to(device)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1).to(device)
            if args.other_model:
                v = v.to(device)
            
            q = question.to(device)
            a = target.to(device)
            
            # Get question classification (OPEN/CLOSED)
            question_logits = question_classifier(q)
            question_pred = question_logits.data.max(1)[1]  # 0=CLOSED, 1=OPEN
            
            # Forward through VQA model
            if args.autoencoder:
                last_output_close, last_output_open, a_close, a_open, decoder = model.forward_classify(
                    v, q, a, question_classifier)
            else:
                last_output_close, last_output_open, a_close, a_open = model.forward_classify(
                    v, q, a, question_classifier)
            
            # Get predictions
            preds_close, preds_open = model.classify(last_output_close, last_output_open)
            
            # Process each sample in batch
            for i in range(len(q)):
                # Decode question
                question_ids = question[i].cpu().numpy()
                question_words = []
                for token_id in question_ids:
                    if token_id != 0 and token_id < len(dictionary.idx2word):
                        question_words.append(dictionary.idx2word[token_id])
                question_text = ' '.join(question_words)
                
                # Get answer prediction
                q_type = question_pred[i].item()
                q_type_name = "OPEN" if q_type == 1 else "CLOSED"
                
                if q_type == 0:  # CLOSED
                    # Find which position this sample is in the close batch
                    close_idx = (answer_target[:i+1] == 0).sum() - 1
                    if close_idx >= 0 and close_idx < len(preds_close):
                        pred_idx = preds_close[close_idx].data.max(0)[1].item()
                        # Get answer from label
                        if pred_idx < len(dataset.label2ans):
                            predicted_answer = dataset.label2ans[pred_idx]
                        else:
                            predicted_answer = f"<unknown_closed_{pred_idx}>"
                    else:
                        predicted_answer = "<no_close_pred>"
                else:  # OPEN
                    # Find which position this sample is in the open batch
                    open_idx = (answer_target[:i+1] == 1).sum() - 1
                    if open_idx >= 0 and open_idx < len(preds_open):
                        pred_idx = preds_open[open_idx].data.max(0)[1].item() + 56  # offset for open answers
                        # Get answer from label
                        if pred_idx < len(dataset.label2ans):
                            predicted_answer = dataset.label2ans[pred_idx]
                        else:
                            predicted_answer = f"<unknown_open_{pred_idx}>"
                    else:
                        predicted_answer = "<no_open_pred>"
                
                # Get ground truth answer
                gt_idx = target[i].data.max(0)[1].item()
                if gt_idx < len(dataset.label2ans):
                    ground_truth = dataset.label2ans[gt_idx]
                else:
                    ground_truth = f"<unknown_gt_{gt_idx}>"
                
                # Store prediction
                pred_dict = {
                    'question_id': batch_idx * args.batch_size + i,
                    'question': question_text,
                    'predicted_answer': predicted_answer,
                    'ground_truth': ground_truth,
                    'question_type_pred': q_type_name,
                    'question_type_gt': answer_type[i],
                    'correct': predicted_answer.lower() == ground_truth.lower()
                }
                predictions.append(pred_dict)
    
    return predictions


def main():
    global args
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

    # Load question classifier
    glove_path = os.path.join(data, 'glove6b_init_300d.npy')
    question_classifier = classify_model(d.ntoken, glove_path).to(device)
    
    classifier_ckpt = os.path.join(root, args.classifier_path)
    print(f"Loading question classifier from: {classifier_ckpt}")
    state = torch.load(classifier_ckpt, map_location=device)
    try:
        question_classifier.load_state_dict(state)
    except:
        question_classifier.load_state_dict(state.get('model_state', state))
    print("Question classifier loaded\n")

    # Load VQA model
    print("Building BAN VQA model...")
    model = BAN_Model(test_dataset, args).to(device)
    
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading VQA model from: {args.model_path}")
        pre_ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(pre_ckpt.get('model_state', pre_ckpt))
        print("VQA model loaded\n")
    else:
        print("WARNING: No trained VQA model checkpoint found!")
        print(f"Looking for: {args.model_path}")
        print("Predictions will use untrained model (random initialization)")
        print("To get meaningful predictions, train the model first using main.py\n")

    # Make predictions
    predictions = predict(model, question_classifier, test_loader, test_dataset, d, device, args.show_samples)
    
    # Calculate accuracy
    correct = sum(1 for p in predictions if p['correct'])
    total = len(predictions)
    accuracy = 100.0 * correct / total
    
    print("\n" + "=" * 80)
    print(f"OVERALL ACCURACY: {accuracy:.2f}% ({correct}/{total} correct)")
    print("=" * 80)
    
    # Show sample predictions
    print(f"\nSHOWING FIRST {args.show_samples} PREDICTIONS:")
    print("=" * 80)
    
    for i, pred in enumerate(predictions[:args.show_samples]):
        status = "CORRECT" if pred['correct'] else "WRONG"
        print(f"\nSample {i + 1} [{status}]")
        print(f"Question: {pred['question']}")
        print(f"Question Type (Predicted): {pred['question_type_pred']}")
        print(f"Ground Truth Answer: {pred['ground_truth']}")
        print(f"Predicted Answer:    {pred['predicted_answer']}")
    
    # Save all predictions
    output_path = os.path.join(root, args.save_predictions)
    with open(output_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'total_samples': total,
            'correct': correct,
            'predictions': predictions
        }, f, indent=2)
    
    print(f"\nAll predictions saved to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
