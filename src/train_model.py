import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from .feature_extractor import extract_features, get_feature_keys
import logging
import sys

def train_model(pdf_path: str, json_path: str, model_output_dir: str) -> dict:
    """
    Train a machine learning model to classify PDF text elements.
    
    Args:
        pdf_path (str): Path to the training PDF file
        json_path (str): Path to the ground truth JSON file
        model_output_dir (str): Directory to save trained model files
        
    Returns:
        dict: Training results and model information
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting model training process...")
    
    # Validate input files
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Extract features from PDF
    logger.info("Extracting features from PDF...")
    all_lines_features = extract_features(pdf_path)
    
    if not all_lines_features:
        raise ValueError("No text features could be extracted from the PDF")
    
    # Load ground truth data
    logger.info("Loading ground truth data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    # Create lookup dictionary for labels
    truth_lookup = {}
    
    # Add title
    if 'title' in ground_truth:
        truth_lookup[ground_truth['title'].strip()] = 'Title'
        logger.info(f"Added title: '{ground_truth['title'].strip()}'")
    
    # Add outline items
    if 'outline' in ground_truth:
        for item in ground_truth['outline']:
            truth_lookup[item['text'].strip()] = item['level']
            logger.info(f"Added {item['level']}: '{item['text'].strip()}'")
    
    logger.info(f"Total ground truth items: {len(truth_lookup)}")
    logger.info(f"Ground truth labels: {set(truth_lookup.values())}")
    
    # Debug: Show some sample text from PDF
    logger.info("Sample PDF text lines:")
    for i, line in enumerate(all_lines_features[:10]):
        logger.info(f"  {i+1}: '{line['text']}'")
    if len(all_lines_features) > 10:
        logger.info(f"  ... and {len(all_lines_features) - 10} more lines")
    
    # Prepare training data
    feature_keys = get_feature_keys()
    X_train = []
    y_train = []
    
    logger.info("Creating training dataset...")
    matched_count = 0
    for line_features in all_lines_features:
        text = line_features["text"].strip()
        label = truth_lookup.get(text, 'Body Text')
        
        if label != 'Body Text':
            matched_count += 1
            logger.info(f"MATCHED: '{text}' -> {label}")
        
        # Extract feature vector
        feature_vector = []
        for key in feature_keys:
            value = line_features.get(key, 0)
            # Convert boolean to int for Random Forest
            if isinstance(value, bool):
                value = int(value)
            feature_vector.append(value)
        
        X_train.append(feature_vector)
        y_train.append(label)
    
    logger.info(f"Matched {matched_count} out of {len(all_lines_features)} text lines with ground truth")
    
    # Debug: Show potential near matches
    if matched_count == 0:
        logger.info("\nNo exact matches found. Checking for potential near matches...")
        ground_truth_texts = set(truth_lookup.keys())
        pdf_texts = set(line['text'].strip() for line in all_lines_features)
        
        logger.info("First 10 ground truth texts:")
        for i, text in enumerate(list(ground_truth_texts)[:10]):
            logger.info(f"  GT: '{text}'")
        
        logger.info("\nFirst 10 PDF texts:")
        for i, text in enumerate(list(pdf_texts)[:10]):
            logger.info(f"  PDF: '{text}'")
    
    if len(X_train) == 0:
        raise ValueError("No training data could be created")
    
    logger.info(f"Training dataset created with {len(X_train)} samples")
    
    # Check label distribution
    label_counts = {}
    for label in y_train:
        label_counts[label] = label_counts.get(label, 0) + 1
    logger.info(f"Label distribution: {label_counts}")
    
    # Encode labels
    logger.info("Encoding labels...")
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    
    # Split data for validation
    if len(X_train) > 10:  # Only split if we have enough data
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
        )
    else:
        X_train_split, X_val_split = X_train, X_train
        y_train_split, y_val_split = y_train_encoded, y_train_encoded
    
    # Train model
    logger.info("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    model.fit(X_train_split, y_train_split)
    
    # Evaluate model
    y_pred = model.predict(X_val_split)
    accuracy = accuracy_score(y_val_split, y_pred)
    
    logger.info(f"Model training complete. Validation accuracy: {accuracy:.3f}")
    
    # Save model artifacts
    logger.info("Saving model artifacts...")
    model_path = os.path.join(model_output_dir, 'heading_model.pkl')
    encoder_path = os.path.join(model_output_dir, 'label_encoder.pkl')
    features_path = os.path.join(model_output_dir, 'feature_keys.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(feature_keys, features_path)
    
    # Create training info
    training_info = {
        'accuracy': float(accuracy),
        'num_samples': len(X_train),
        'num_features': len(feature_keys),
        'classes': list(encoder.classes_),
        'label_distribution': label_counts,
        'feature_keys': feature_keys
    }
    
    logger.info("Training completed successfully!")
    return training_info
