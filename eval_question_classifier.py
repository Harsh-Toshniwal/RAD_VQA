# -*- coding: utf-8 -*-
"""
Evaluation script for the question type classifier
Loads type_classifier.pth and evaluates on test set
"""

import os
import torch
from torch.utils.data import DataLoader
import utils
from dataset_RAD import Dictionary, VQAFeatureDataset
from classify_question import classify_model, evaluate as evaluate_cls


def parse_args():
    """Parse command line arguments (reusing main.py's args structure)"""
    import argparse
    parser = argparse.ArgumentParser(description="Med VQA Classifier Evaluation")
    parser.add_argument('--seed', type=int, default=5, help='random seed. default:5')
    parser.add_argument('--gpu', type=int, default=0, help='use gpu device. default:0')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for evaluation')
    
    # Vision flags (needed by dataset)
    parser.add_argument('--v_dim', default=64, type=int, help='visual feature dim')
    parser.add_argument('--autoencoder', action='store_true', default=True)
    parser.add_argument('--maml', action='store_true', default=True)
    parser.add_argument('--other_model', action='store_true', default=False)
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights')
    
    args = parser.parse_args()
    return args


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
    if not os.path.exists(dict_path):
        print(f"Dictionary not found at {dict_path}")
        print("Please run tools/create_dictionary.py first")
        return
    
    d = Dictionary.load_from_file(dict_path)
    print(f"Dictionary loaded: {d.ntoken} tokens")

    # Build test dataset/loader
    print("Loading test dataset...")
    test_dataset = VQAFeatureDataset('test', args, d, dataroot=data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)
    print(f"Test dataset size: {len(test_dataset)}")

    # Build classifier model
    glove_path = os.path.join(data, 'glove6b_init_300d.npy')
    if not os.path.exists(glove_path):
        print(f"Warning: GloVe embeddings not found at {glove_path}")
        print("Model may not initialize correctly")
    
    net = classify_model(d.ntoken, glove_path)
    net = net.to(device)

    # Load checkpoint
    ckpt_path = os.path.join(root, 'saved_models', 'type_classifier.pth')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    print(f"Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats
    try:
        net.load_state_dict(state)
    except Exception as e:
        print(f"Direct load failed, trying 'model_state' key: {e}")
        try:
            net.load_state_dict(state.get('model_state', state))
        except Exception as e2:
            print(f"Failed to load checkpoint: {e2}")
            return
    
    print("Model loaded successfully")

    # Create logger
    log_dir = os.path.join(root, 'log_eval')
    utils.create_dir(log_dir)
    logger = utils.Logger(os.path.join(log_dir, 'eval_classifier.log')).get_logger()
    
    logger.info("=" * 60)
    logger.info("Question Type Classifier Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Device: {device}")

    # Evaluate
    print("\nEvaluating on test set...")
    acc = evaluate_cls(net, test_loader, logger, device)
    
    print("=" * 60)
    print(f"Test Accuracy: {acc:.2f}%")
    print("=" * 60)
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
