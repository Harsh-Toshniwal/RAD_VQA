# -*- coding: utf-8 -*-

"""
Utility functions for analyzing test predictions JSON
"""

import json

def load_predictions(filepath='test_predictions.json'):
    """Load predictions from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_statistics(data):
    """Extract and display statistics"""
    stats = data['statistics']
    print("="*60)
    print("PREDICTION STATISTICS")
    print("="*60)
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Correct Predictions: {stats['correct_predictions']}")
    print(f"Overall Accuracy: {stats['overall_accuracy']*100:.2f}%")
    print(f"\nCLOSED Questions:")
    print(f"  Total: {stats['closed_count']}")
    print(f"  Correct: {stats['closed_correct']}")
    print(f"  Accuracy: {stats['closed_accuracy']*100:.2f}%")
    print(f"\nOPEN Questions:")
    print(f"  Total: {stats['open_count']}")
    print(f"  Correct: {stats['open_correct']}")
    print(f"  Accuracy: {stats['open_accuracy']*100:.2f}%")
    print("="*60)

def get_incorrect_predictions(data):
    """Get all incorrect predictions"""
    return [p for p in data['predictions'] if not p['is_correct']]

def get_correct_predictions(data):
    """Get all correct predictions"""
    return [p for p in data['predictions'] if p['is_correct']]

def get_high_confidence_wrong(data, threshold=0.9):
    """Get high-confidence wrong predictions"""
    return [p for p in data['predictions'] 
            if not p['is_correct'] and p['predicted_confidence'] > threshold]

def get_by_question_type(data, q_type='CLOSED'):
    """Get predictions by question type"""
    return [p for p in data['predictions'] if p['question_type'] == q_type]

def get_by_keyword(data, keyword):
    """Get predictions containing keyword in question"""
    return [p for p in data['predictions'] if keyword.lower() in p['question'].lower()]

def print_prediction(pred):
    """Pretty print a prediction record"""
    print("="*60)
    print(f"Sample #{pred['sample_idx']}")
    print(f"Question: {pred['question']}")
    print(f"Type: {pred['question_type']} | Answer Type: {pred['answer_type']}")
    print(f"\nPredicted: {pred['predicted_answer']}")
    print(f"Confidence: {pred['predicted_confidence']:.4f}")
    print(f"Ground Truth: {pred['ground_truth_answer']}")
    print(f"Correct: {'[YES]' if pred['is_correct'] else '[NO]'}")
    print("="*60)

def analyze_errors(data, top_n=10):
    """Analyze most common error patterns"""
    incorrect = get_incorrect_predictions(data)
    
    print("\n" + "="*60)
    print(f"ANALYSIS: Top {top_n} Incorrect Predictions")
    print("="*60)
    
    # Sort by confidence (high confidence mistakes are more interesting)
    sorted_errors = sorted(incorrect, key=lambda x: x['predicted_confidence'], reverse=True)
    
    for i, pred in enumerate(sorted_errors[:top_n], 1):
        print(f"\n{i}. {pred['question']}")
        print(f"   Predicted: {pred['predicted_answer']} (confidence: {pred['predicted_confidence']:.4f})")
        print(f"   Ground Truth: {pred['ground_truth_answer']}")

def export_subset(data, output_file, condition_func):
    """Export subset of predictions based on condition"""
    subset = [p for p in data['predictions'] if condition_func(p)]
    
    subset_data = {
        'model_info': data['model_info'],
        'statistics': {
            'total_samples': len(subset),
            'correct_predictions': sum(1 for p in subset if p['is_correct']),
            'overall_accuracy': sum(1 for p in subset if p['is_correct']) / len(subset) if subset else 0
        },
        'predictions': subset
    }
    
    with open(output_file, 'w') as f:
        json.dump(subset_data, f, indent=2)
    
    print(f"Exported {len(subset)} predictions to {output_file}")

# Example usage
if __name__ == '__main__':
    # Load predictions
    data = load_predictions()
    
    # Show statistics
    get_statistics(data)
    
    # Get some examples
    print("\n[1] Example: Incorrect Predictions")
    incorrect = get_incorrect_predictions(data)
    if incorrect:
        print_prediction(incorrect[0])
    
    print("\n[2] Example: High-confidence Wrong Predictions")
    high_conf_wrong = get_high_confidence_wrong(data, threshold=0.95)
    print(f"Found {len(high_conf_wrong)} high-confidence mistakes")
    if high_conf_wrong:
        print_prediction(high_conf_wrong[0])
    
    print("\n[3] Example: CLOSED Questions Only")
    closed = get_by_question_type(data, 'CLOSED')
    print(f"CLOSED Questions: {len([p for p in closed if p['is_correct']])}/{len(closed)} correct")
    
    print("\n[4] Example: Questions about specific topics")
    aortic_q = get_by_keyword(data, 'aortic')
    print(f"Questions about 'aortic': {len(aortic_q)} found")
    if aortic_q:
        print_prediction(aortic_q[0])
    
    # Analyze errors
    analyze_errors(data, top_n=5)
    
    # Export examples
    print("\n[5] Exporting subsets...")
    export_subset(data, 'incorrect_predictions.json', lambda p: not p['is_correct'])
    export_subset(data, 'high_confidence_predictions.json', lambda p: p['predicted_confidence'] > 0.95)
