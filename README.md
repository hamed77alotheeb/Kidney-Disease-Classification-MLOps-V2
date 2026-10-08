# Kidney Disease Classification MLOps

Deep learning project for four-class kidney CT image classification.

## Classes

- Cyst
- Normal
- Stone
- Tumor

## Dataset

The cleaned dataset contains:

- 11,929 unique images
- Train: 8,348
- Validation: 1,788
- Test: 1,793

Exact duplicate image copies were removed before creating the final
training, validation, and test split.

The final split was checked for exact SHA-256 hash overlap:

- Train ∩ Validation = 0
- Train ∩ Test = 0
- Validation ∩ Test = 0

## Environment

- Python 3.8
- TensorFlow 2.10.1
- Keras 2.10.0
- NumPy 1.24.4
- Pandas 2.0.3
- Scikit-learn 1.3.2

GPU acceleration is available through the NVIDIA GPU environment.

## Development Stages

1. Dataset verification
2. Baseline model
3. Model training
4. Model evaluation
5. Inference
6. API
7. Testing
8. Docker
9. GitHub CI/CD

## Important

The final model accuracy will be calculated on a separate test set
that is not used during model training.

Confidence scores will represent model probabilities and will not
be treated as a guarantee of medical diagnosis.
