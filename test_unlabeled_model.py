import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_unlabeled_cnn import (  # noqa: E402
    CenterOfMassAlignLayer,
    HorizontalFlipLayer,
    RESEARCH_DIR,
    generate_synthetic_data,
)

TEST_IMAGES_PER_CLASS = 500


def preview_test_samples(X_test, y_test, aligned_weights, y_pred, num_samples=5, save_path="test_sample_preview.png"):
    """
    Displays generated images alongside their ground-truth class,
    aligned weight score [-1, +1], and thresholded prediction.
    """
    n = min(num_samples, len(X_test))
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.2))

    for idx in range(n):
        img = X_test[idx, :, :, 0]
        true_lbl = "S (+1)" if y_test[idx] > 0 else "Anti-S (-1)"
        pred_lbl = "S (+1)" if y_pred[idx] > 0 else "Anti-S (-1)"
        is_correct = (y_test[idx] == y_pred[idx])
        title_color = "darkgreen" if is_correct else "crimson"

        axes[idx].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[idx].set_title(
            f"Test #{idx}\nTrue: {true_lbl}\nPred: {pred_lbl}\nW = {aligned_weights[idx]:+.3f}",
            fontsize=8.5,
            color=title_color
        )
        axes[idx].axis("off")

    plt.suptitle("Generated Sample Preview (Ground Truth vs. Model Output)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓ Saved 5-image test preview -> '{save_path}'")
    plt.show()


# =====================================================================
# MAIN EVALUATION & VISUALIZATION PIPELINE
# =====================================================================
def main():
    model_path = os.path.join(RESEARCH_DIR, "s_anti_s_blind_tanh.keras")
    output_plot_path = os.path.join(SCRIPT_DIR, "chiral_evaluation_results.png")
    preview_path = os.path.join(SCRIPT_DIR, "test_sample_preview.png")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Run 'train_unlabeled_cnn.py' first."
        )

    print("1. Loading the trained anti-symmetric model and generating evaluation images...")
    custom_objs = {
        "HorizontalFlipLayer": HorizontalFlipLayer,
        "CenterOfMassAlignLayer": CenterOfMassAlignLayer,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)

    X_test, y_test = generate_synthetic_data(
        num_samples_per_class=TEST_IMAGES_PER_CLASS,
        target_size=36,
        seed=43,
    )
    if len(X_test) == 0:
        raise ValueError("No generated images were found for evaluation.")

    num_s = int(np.sum(np.isclose(y_test, 1.0)))
    num_anti_s = int(np.sum(np.isclose(y_test, -1.0)))
    print(
        f"   ✓ Loaded {len(X_test)} generated images (shape {X_test.shape[1:]})"
        f" -> {num_s} S (+1) | {num_anti_s} Anti-S (-1)"
    )
    print("   Note: These are newly generated evaluation images, not saved training images.")

    # =====================================================================
    # 2. INFERENCE & MIRROR ANTI-SYMMETRY VERIFICATION
    # =====================================================================
    raw_weights = model.predict(X_test, batch_size=128, verbose=0).reshape(-1)
    mirror_weights = model.predict(np.flip(X_test, axis=2), batch_size=128, verbose=0).reshape(-1)

    max_asymmetry = float(np.max(np.abs(raw_weights + mirror_weights)))
    print(f"   ✓ Model Mirror Anti-Symmetry Check: Max |W(X) + W(flip(X))| = {max_asymmetry:.2e}")

    print("\nSample Raw Weights Assigned by Model (First 10 Test Images):")
    print("-" * 58)
    print(f"{'Index':<8} | {'True Label':<14} | {'Assigned Weight (-1 to +1)':<28}")
    print("-" * 58)
    for idx in range(min(10, len(X_test))):
        true_str = "S (+1.0)" if y_test[idx] > 0 else "Anti-S (-1.0)"
        print(f"{idx:<8} | {true_str:<14} | {raw_weights[idx]:+.4f}")
    print("-" * 58)

    # =====================================================================
    # 3. PARITY ALIGNMENT & THRESHOLD CLASSIFICATION (0.0 BOUNDARY)
    # =====================================================================
    initial_preds = np.where(raw_weights >= 0.0, 1.0, -1.0)
    initial_acc = accuracy_score(y_test, initial_preds)

    if initial_acc < 0.5:
        print("\nℹ️ Parity Detection: Blind model clustered S -> -1.0 and Anti-S -> +1.0.")
        print("   Flipping sign for ground-truth alignment...")
        aligned_weights = -raw_weights
    else:
        print("\nℹ️ Parity Detection: Blind model clustered S -> +1.0 and Anti-S -> -1.0.")
        aligned_weights = raw_weights

    y_pred = np.where(aligned_weights >= 0.0, 1.0, -1.0)
    final_acc = accuracy_score(y_test, y_pred)
    print(f"✓ Generated-image evaluation accuracy: {final_acc * 100:.2f}%\n")

    # Visual inspection of 5 generated samples with predictions
    preview_test_samples(
        X_test, y_test, aligned_weights, y_pred,
        num_samples=5, save_path=preview_path
    )

    # =====================================================================
    # 4. CONFUSION MATRIX VISUALIZATION
    # =====================================================================
    fig, ax = plt.subplots(figsize=(7, 6))

    cm = confusion_matrix(y_test, y_pred, labels=[-1.0, 1.0])
    im = ax.imshow(cm, cmap="Purples")
    ax.set_title(f"Confusion Matrix (n={len(X_test)}, Threshold=0.0)\nAccuracy: {final_acc * 100:.2f}%")
    plt.colorbar(im, ax=ax)

    tick_labels = ["Anti-S (-1)", "S (+1)"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(tick_labels)

    cmap = im.get_cmap()
    norm = im.norm
    for i in range(2):
        for j in range(2):
            red, green, blue, _ = cmap(norm(cm[i, j]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="black" if luminance > 0.5 else "white",
                fontsize=14, fontweight="bold"
            )

    ax.set_ylabel("Actual Ground-Truth Class")
    ax.set_xlabel("Predicted Class (Weight < 0 vs >= 0)")

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150)
    print(f"✓ Saved confusion matrix to '{output_plot_path}'")
    plt.show()

    print("\nDetailed Classification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=[-1.0, 1.0],
        target_names=["Anti-S (-1)", "S (+1)"],
        zero_division=0
    ))


if __name__ == "__main__":
    main()