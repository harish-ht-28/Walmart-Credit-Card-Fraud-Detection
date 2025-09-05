import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix, roc_curve,
                           precision_recall_curve, classification_report)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, BatchNormalization,
                                   Dropout, LayerNormalization, LeakyReLU)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                      TensorBoard)
from tensorflow.keras.metrics import (Precision, Recall, AUC)
from sklearn.model_selection import GridSearchCV

# Configuration
import numpy as np
from tensorflow.keras.metrics import Precision, Recall, AUC


CONFIG = {
    'random_state': 42,  # Reproducibility
    'test_size': 0.2,    # 20% test split
    'validation_split': 0.2,  # 20% of training data for validation
    'batch_size': 512,   # Suitable for large datasets
    'epochs': 150,       # Sufficient for convergence
    'class_weight': {0: 1.5, 1: 0.8},  # Adjusted weights: moderate non-fraud, higher fraud weight
    'under_sampling_ratio': 0.5,  # Undersample non-fraud to 50% of fraud cases for balance
    'threshold_search_range': np.linspace(0.2, 0.5, 50),  # Expanded range, starting higher to balance recall/precision
    'metrics': [
        'accuracy',
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc_roc'),
        AUC(name='auc_pr', curve='PR'),
        'f1_score'
    ]
}

def load_and_preprocess_data():
    """Load and preprocess data with comprehensive feature engineering"""
    print("⏳ Loading and preprocessing data...")

    try:
        df = pd.read_csv("/content/drive/MyDrive/Hackathon Dataset/improved_dataset.csv")

        # Convert column names to strings
        df.columns = df.columns.astype(str)

        # Identify datetime columns
        datetime_cols = []
        for col in df.columns:
            if isinstance(col, str) and ('date' in col.lower() or 'time' in col.lower()):
                datetime_cols.append(col)

        # Process datetime features
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_is_weekend'] = (df[f'{col}_dayofweek'] >= 5).astype(int)
            df.drop(col, axis=1, inplace=True)

        # Process categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = pd.factorize(df[col])[0]

        # Handle other string columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"Warning: Dropping non-numeric columns: {list(non_numeric_cols)}")
            df = df.drop(columns=non_numeric_cols)

        # Ensure target exists
        if 'is_fraud' not in df.columns:
            raise ValueError("Target column 'is_fraud' not found")

        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def build_enhanced_model(input_dim):
    """Build fraud detection model with improved architecture"""
    inputs = Input(shape=(input_dim,))

    # Feature processing block
    x = Dense(128, activation=None, kernel_regularizer=l2(0.01))(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Feature extraction block
    x = Dense(64, activation=None, kernel_regularizer=l2(0.005))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    # Prediction block
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=CONFIG['metrics']
    )
    return model

def train_validate_model(X_train, y_train):
    """Train model with class balancing and proper validation"""
    print("\n🏗 Training fraud detection model...")

    # Apply conservative under-sampling
    under_sampler = RandomUnderSampler(
        sampling_strategy=CONFIG['under_sampling_ratio'],
        random_state=CONFIG['random_state']
    )
    X_res, y_res = under_sampler.fit_resample(X_train, y_train)

    # Compute class weights
    class_weights = None
    if CONFIG['class_weight']:
        classes = np.unique(y_res)
        weights = compute_class_weight('balanced', classes=classes, y=y_res)
        class_weights = dict(zip(classes, weights))

    # Build model
    model = build_enhanced_model(X_res.shape[1])

    # Callbacks
    callbacks = [
        EarlyStopping(
            patience=10,
            monitor='val_auc_pr',
            mode='max',
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        TensorBoard(log_dir='./logs')
    ]

    # Train model
    history = model.fit(
        X_res, y_res,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        validation_split=CONFIG['validation_split'],
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    return model, history

def optimize_classification_threshold(model, X_val, y_val):
    """Find optimal threshold balancing precision and recall"""
    y_proba = model.predict(X_val).flatten()
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

    # Find threshold that maximizes F1 score
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]

    print(f"\n⚖ Optimal threshold: {best_threshold:.4f}")
    return best_threshold

def evaluate_model_performance(model, X_test, y_test, threshold):
    """Comprehensive model evaluation with visualization"""
    y_proba = model.predict(X_test).flatten()
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'conf_matrix': confusion_matrix(y_test, y_pred)
    }

    # Print metrics
    print("\n📊 Model Evaluation:")
    print(f"Threshold: {threshold:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

    # Plot metrics
    plot_performance_metrics(y_test, y_proba, threshold, metrics['conf_matrix'])

def plot_performance_metrics(y_true, y_proba, threshold, conf_matrix):
    """Visualize key performance metrics"""
    plt.figure(figsize=(15, 12))

    # Confusion Matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Fraud', 'Fraud'],
        yticklabels=['Not Fraud', 'Fraud']
    )
    plt.title(f'Confusion Matrix\n(Threshold: {threshold:.2f})')

    # ROC Curve
    plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Precision-Recall Curve
    plt.subplot(2, 2, 3)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    # Threshold Analysis
    plt.subplot(2, 2, 4)
    thresholds = np.linspace(0.1, 0.99, 50)
    precisions = []
    recalls = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.axvline(x=threshold, color='r', linestyle='--', label='Selected Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Analysis')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()

        # Prepare features and target
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=CONFIG['test_size'],
            random_state=CONFIG['random_state'],
            stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model, history = train_validate_model(X_train_scaled, y_train)

        # Optimize threshold
        # Use a subset of the training data for threshold optimization
        subset_size = min(10000, len(X_train_scaled)) # Use up to 10000 samples
        threshold = optimize_classification_threshold(
            model,
            X_train_scaled[:subset_size],
            y_train[:subset_size]
        )

        # Evaluate model
        evaluate_model_performance(
            model,
            X_test_scaled,
            y_test,
            threshold
        )

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
