# Unlabeled Chiral CNN
Classifies S or anti S trained using unlabeled data.

## Requirements

- TensorFlow 2.16 or later
- NumPy
- Pillow
- Matplotlib
- scikit-learn (evaluation script only)


Each run generates 5,000 randomized S images and their 5,000 horizontal mirrors at 36 × 36 pixels. The labeled pairs are shuffled and the labels are not supplied to the trainer. The CNN is trained for 12 epochs with batches of 64 using the loss function 
$$
\mathcal{L}_{\text{unlabeled}} = \left( \frac{\vert{}P(X_{\text{orig}}) - P(X_{\text{mirror}})\vert{}}{2} - 0.5 \right)^2
$$

On completion, it saves:

- `../s_anti_s_blind_tanh.keras` — trained model in the parent
  directory.
- `train_sample_preview.png` — preview saved to the current working directory.

## Testing

After training, run:

```bash
python test_unlabeled_model.py
```

The evaluation script generates 500 S images and 500 mirrored Anti-S images, then evaluates the saved model. It reports mirror anti-symmetry, accuracy, a classification report, and a confusion matrix.