"""
Cluster-aware P-K batch sampler for triplet loss.

Guarantees:
    - Each batch contains P dogs, K images per dog
    - For each dog, K images are drawn from DISTINCT clusters (sessions)
    - Every image used at most once per dog per batch

Input CSV columns: image, dog_id, cluster_id, abs_path
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.biometric.augmentation import augment_train, augment_eval, IMG_SIZE


SEED = 42
P_DOGS_PER_BATCH = 8
K_IMAGES_PER_DOG = 4


def load_image(path):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return img


def preprocess(img):
    return preprocess_input(img)


class PKBatchGenerator:
    """
    Cluster-aware P-K sampler.

    For each dog in the batch:
      - Shuffle clusters
      - Take images from each cluster one at a time, moving to the next
        cluster after taking 1-2 from the current
      - Never take the same image twice within one batch's dog-portion

    If a dog's total unique images < K, we pad with duplicates but the
    caller can filter those dogs out. Training CSV only contains dogs
    with >= 2 clusters.
    """

    def __init__(self, df, dog_to_idx, P=8, K=4, augment=True, seed=42):
        self.P = P
        self.K = K
        self.augment = augment
        self.dog_to_idx = dog_to_idx
        self.rng = np.random.default_rng(seed)

        # Build: dog_id -> list of (cluster_id, [paths])
        self.dog_to_clusters = {}
        for dog_id, group in df.groupby("dog_id"):
            clusters = []
            for cid, sub in group.groupby("cluster_id"):
                clusters.append((int(cid), sub["abs_path"].tolist()))
            self.dog_to_clusters[int(dog_id)] = clusters

        # Only keep dogs with >= 2 clusters AND >= K total images
        self.dogs = []
        for d, clusters in self.dog_to_clusters.items():
            total = sum(len(paths) for _, paths in clusters)
            if len(clusters) >= 2 and total >= self.K:
                self.dogs.append(d)

        self.batch_size = P * K

        # Stats
        self.total_images_sampled = 0
        self.repeat_occurrences = 0

    def __len__(self):
        return 1

    def _sample_paths_for_dog(self, dog):
        """
        Return K unique paths for this dog, drawn from different clusters.
        Never returns the same path twice unless the dog genuinely has
        fewer than K unique images (which we already filter out).
        """
        clusters = self.dog_to_clusters[dog]
        # shuffle cluster order
        order = self.rng.permutation(len(clusters))

        paths = []
        # round-robin across clusters
        used = set()
        # build a per-cluster iterator queue
        cluster_queues = []
        for i in order:
            cid, imgs = clusters[i]
            shuffled = list(self.rng.permutation(imgs))
            cluster_queues.append((cid, shuffled, 0))

        # Round 1: take 1 from each cluster
        for cid, imgs, idx in cluster_queues:
            if len(paths) >= self.K:
                break
            paths.append(imgs[idx])
            used.add(imgs[idx])
            cluster_queues[cluster_queues.index((cid, imgs, idx))] = (cid, imgs, idx + 1)

        # Round 2+: take more from clusters with remaining images
        while len(paths) < self.K:
            progress = False
            for i, (cid, imgs, idx) in enumerate(cluster_queues):
                if len(paths) >= self.K:
                    break
                if idx < len(imgs):
                    paths.append(imgs[idx])
                    used.add(imgs[idx])
                    cluster_queues[i] = (cid, imgs, idx + 1)
                    progress = True
            if not progress:
                break

        return paths

    def sample_batch(self):
        chosen_dogs = self.rng.choice(self.dogs, size=self.P, replace=False)

        paths = []
        labels = []

        for dog in chosen_dogs:
            dog_paths = self._sample_paths_for_dog(dog)

            # Safety check: ensure no dupes
            if len(set(dog_paths)) < len(dog_paths):
                self.repeat_occurrences += len(dog_paths) - len(set(dog_paths))

            self.total_images_sampled += len(dog_paths)
            for p in dog_paths:
                paths.append(p)
                labels.append(self.dog_to_idx[dog])

        # load + augment
        imgs = []
        for p in paths:
            img = load_image(p)
            if self.augment:
                img = augment_train(img)
            else:
                img = augment_eval(img)
            img = preprocess(img)
            imgs.append(img)

        imgs = tf.stack(imgs)
        labels = tf.constant(labels, dtype=tf.int32)
        return imgs, labels

    def stats(self):
        if self.total_images_sampled == 0:
            return {"repeat_rate": 0.0}
        return {
            "total_images_sampled": self.total_images_sampled,
            "repeat_occurrences": self.repeat_occurrences,
            "repeat_rate": self.repeat_occurrences / self.total_images_sampled,
        }


def make_pk_dataset(df, dog_to_idx, P=8, K=4, augment=True,
                    batches_per_epoch=50, seed=42):
    gen = PKBatchGenerator(df, dog_to_idx, P=P, K=K, augment=augment, seed=seed)

    output_signature = (
        tf.TensorSpec(shape=(P * K, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(P * K,), dtype=tf.int32),
    )

    def generator():
        for _ in range(batches_per_epoch):
            yield gen.sample_batch()

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, gen