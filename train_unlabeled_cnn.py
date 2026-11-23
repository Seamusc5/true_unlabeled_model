import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

RESEARCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# CUSTOM SERIALIZABLE LAYERS (SAFE IN KERAS 3)
# =====================================================================
@tf.keras.utils.register_keras_serializable(package="ChiralModels")
class HorizontalFlipLayer(layers.Layer):
    def call(self, inputs):
        return tf.image.flip_left_right(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable(package="ChiralModels")
class CenterOfMassAlignLayer(layers.Layer):
    """
    Centers each image's intensity center-of-mass on the canvas so horizontal
    translation cannot be used as a shortcut over chiral shape.
    """
    def call(self, inputs):
        shape = tf.shape(inputs)
        h = tf.cast(shape[1], tf.float32)
        w = tf.cast(shape[2], tf.float32)

        y_grid = tf.reshape(tf.range(h, dtype=tf.float32), [1, shape[1], 1, 1])
        x_grid = tf.reshape(tf.range(w, dtype=tf.float32), [1, 1, shape[2], 1])

        total_mass = tf.reduce_sum(inputs, axis=[1, 2, 3], keepdims=True) + 1e-6
        center_y = tf.reduce_sum(inputs * y_grid, axis=[1, 2, 3], keepdims=True) / total_mass
        center_x = tf.reduce_sum(inputs * x_grid, axis=[1, 2, 3], keepdims=True) / total_mass

        target_y = (h - 1.0) / 2.0
        target_x = (w - 1.0) / 2.0

        dy = tf.cast(tf.round(target_y - center_y), tf.int32)
        dx = tf.cast(tf.round(target_x - center_x), tf.int32)

        def shift_single(args):
            img, sy, sx = args
            return tf.roll(img, shift=[sy[0, 0, 0], sx[0, 0, 0]], axis=[0, 1])

        return tf.map_fn(shift_single, (inputs, dy, dx), fn_output_signature=tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape


# =====================================================================
# BLOCK 1: GENERATE UNLABELED IMAGES IN MEMORY
# =====================================================================
def generate_synthetic_data(num_samples_per_class=5000, target_size=36, seed=42):
    """
    Generates balanced S and mirrored Anti-S image arrays and their labels
    in memory, without writing image files or creating directories.
    """
    rng = random.Random(seed)
    font_path = "/usr/share/fonts/truetype/noto/NotoSans-Italic.ttf"
    fonts = {}
    s_images = []
    anti_s_images = []

    def to_array(image):
        gray = image.convert("L")
        if gray.size != (target_size, target_size):
            gray = gray.resize((target_size, target_size), Image.Resampling.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        if np.mean(arr) > 0.5:
            arr = 1.0 - arr
        return arr[..., np.newaxis]

    for _ in range(num_samples_per_class):
        font_size = rng.randint(10, 23)
        angle = rng.randint(0, 360)
        if font_size not in fonts:
            try:
                fonts[font_size] = ImageFont.truetype(font_path, font_size)
            except OSError:
                fonts[font_size] = ImageFont.load_default()

        canvas = Image.new("RGBA", (35, 35), color="white")
        ImageDraw.Draw(canvas).text(
            (10, 10), "S", font=fonts[font_size], fill=(0, 0, 0)
        )
        rotated_s = canvas.rotate(angle, expand=False, fillcolor="white")
        s_images.append(to_array(rotated_s))
        anti_s_images.append(to_array(ImageOps.mirror(rotated_s)))

    X = np.concatenate((
        np.stack(s_images),
        np.stack(anti_s_images),
    ), axis=0)
    y = np.concatenate((
        np.ones(num_samples_per_class, dtype=np.float32),
        -np.ones(num_samples_per_class, dtype=np.float32),
    ))
    return X, y


def preview_training_samples(X_train, num_samples=5, save_path="train_sample_preview.png"):
    """
    Saves and displays 5 unlabeled training images alongside their mirror flips.
    """
    n = min(num_samples, len(X_train))
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.0))

    for idx in range(n):
        orig_img = X_train[idx, :, :, 0]
        mirror_img = np.fliplr(orig_img)

        axes[0, idx].imshow(orig_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, idx].set_title(f"Train #{idx}\n(s_generator)", fontsize=9)
        axes[0, idx].axis("off")

        axes[1, idx].imshow(mirror_img, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, idx].set_title(f"Mirror #{idx}\n(Internal Flip)", fontsize=9)
        axes[1, idx].axis("off")

    plt.suptitle("Generated Training Samples", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"   ✓ Saved 5-image training preview -> '{save_path}'")
    plt.show()


# =====================================================================
# BLOCK 2: STRICTLY ANTI-SYMMETRIC CNN + CONSISTENCY SELF-SUPERVISION
# =====================================================================
def build_antisymmetric_cnn(input_shape=(36, 36, 1)):
    feature_scorer = models.Sequential([
        layers.Conv2D(32, (5, 5), padding="same", activation="tanh", input_shape=input_shape),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", activation="tanh"),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", activation="tanh"),
        layers.Flatten(),
        layers.Dense(64, activation="tanh"),
        layers.Dense(1, use_bias=False)
    ], name="feature_scorer")

    inp = layers.Input(shape=input_shape)
    centered = CenterOfMassAlignLayer(name="center_of_mass")(inp)
    centered_mirror = HorizontalFlipLayer(name="mirror_flip")(centered)

    score_orig = feature_scorer(centered)
    score_mirror = feature_scorer(centered_mirror)

    diff_logit = layers.Subtract(name="antisymmetric_logit")([score_orig, score_mirror])
    signed_output = layers.Activation("tanh", name="signed_weight_output")(diff_logit)

    return models.Model(inputs=inp, outputs=signed_output, name="Exact_AntiSymmetric_Chiral_CNN")


class BlindChiralTrainer(tf.keras.Model):
    def __init__(self, antisym_cnn):
        super().__init__()
        self.cnn = antisym_cnn
        # flip followed by a random rotation = mirror about a random axis
        self.random_rotation = layers.RandomRotation(0.5, fill_mode="constant", fill_value=0.0)
        self.loss_tracker = tf.keras.metrics.Mean(name="blind_loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, X_unlabeled):
        # Chirality is odd under EVERY reflection, not just the horizontal flip.
        # Oddness under one fixed flip is also satisfied by "leans / vs \" (tilt),
        # which is what the network learns otherwise.
        X_mirror = self.random_rotation(tf.image.flip_left_right(X_unlabeled), training=True)

        with tf.GradientTape() as tape:
            P_orig = self.cnn(X_unlabeled, training=True)
            P_mirror = self.cnn(X_mirror, training=True)
            # maximise (S(x) - S(Mx))^2; the old line *minimised* it, collapsing S to 0
            blind_loss = -tf.reduce_mean(tf.square((P_orig - P_mirror) / 2.0))

        grads = tape.gradient(blind_loss, self.cnn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.cnn.trainable_variables))

        self.loss_tracker.update_state(blind_loss)
        return {"loss": self.loss_tracker.result()}

    def call(self, inputs):
        return self.cnn(inputs)


# =====================================================================
# BLOCK 3: EXECUTION
# =====================================================================
if __name__ == "__main__":
    model_output_path = os.path.join(RESEARCH_DIR, "s_anti_s_blind_tanh.keras")
    if os.path.exists(model_output_path):
        os.remove(model_output_path)

    print("1. Generating 5,000 S and 5,000 Anti-S images in memory...")
    X_train_unlabeled, _ = generate_synthetic_data(
        num_samples_per_class=5000, target_size=36
    )
    shuffle_idx = np.random.permutation(len(X_train_unlabeled))
    X_train_unlabeled = X_train_unlabeled[shuffle_idx]
    print(
        f"   ✓ Generated {len(X_train_unlabeled)} unlabeled images "
        f"| shape: {X_train_unlabeled.shape} "
        f"| pixel range: [{X_train_unlabeled.min():.2f}, {X_train_unlabeled.max():.2f}]"
    )

    preview_training_samples(X_train_unlabeled, num_samples=5, save_path="train_sample_preview.png")

    antisym_cnn = build_antisymmetric_cnn(input_shape=X_train_unlabeled.shape[1:])
    trainer = BlindChiralTrainer(antisym_cnn)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    print("\n3. Starting Anti-Symmetric Blind Self-Supervised Training...")
    trainer.fit(X_train_unlabeled, epochs=12, batch_size=64)

    sample_batch = X_train_unlabeled[:100]
    w1 = antisym_cnn.predict(sample_batch, verbose=0)
    w2 = antisym_cnn.predict(np.flip(sample_batch, axis=2), verbose=0)
    print(f"\n✓ Pre-Save Anti-Symmetry Check: Max |W(X) + W(flip(X))| = {np.max(np.abs(w1 + w2)):.2e}")

    antisym_cnn.save(model_output_path)
    print(f"✓ Clean anti-symmetric model saved to '{model_output_path}'")