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
\mathcal{L}_{\text{unlabeled}} = -\left\langle \left( \frac{P(X) - P(M_\phi X)}{2} \right)^2 \right\rangle
$$

where $M_\phi$ is a mirror about a **random axis** (horizontal flip followed by a random rotation). A mirror about a single fixed axis is not enough. See [docs/unlabeled_chirality_investigation.md](docs/unlabeled_chirality_investigation.md) for why.

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
