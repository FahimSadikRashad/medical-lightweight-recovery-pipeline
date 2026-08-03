"""Baseline 2 -- Paper A's method: retrain with MedMNIST-C augmentation.

Same architecture as Baseline 1, but the official corruptions are injected
during training. Answers: how much of the collapse does augmentation alone
recover? That number is the bar the recovery module must beat or complement.

Mechanism difference from Baseline 1:
  evaluation -> corrupt(img, severity)   fixed severity, reproducible test set
  training   -> augmenter()              random corruption AND severity per sample

Note this baseline requires retraining the classifier, which is exactly what
the recovery approach is designed to avoid -- so a fair reading is "different
mechanism", not "we beat it".

    python scripts/02_baseline2_augmented.py
"""
from _common import balanced, config, data, engine, models, parse_args, setup, store

import pandas as pd

from robustmed.corruptions import augmenter

if __name__ == "__main__":
    args = parse_args()
    train, val, test, info, n_classes = setup(args)
    if args.epochs:
        config.CLF_EPOCHS = args.epochs

    aug_model = models.Classifier(n_classes).to(engine.DEVICE)

    print("\n== training with MedMNIST-C augmentation ==")
    engine.train_classifier(
        aug_model,
        data.loader(data.Clean(train, pil_transform=augmenter()), shuffle=True),
        data.loader(data.Clean(val)),          # validate on clean
        config.BASELINE2_CKPT,
        epochs=config.CLF_EPOCHS,
    )

    print("\n== same conditions as Baseline 1 ==")
    aug_model.freeze()
    results = engine.eval_conditions(None, aug_model, test)
    store.save(store.BASELINE2, results)

    # Compare against Baseline 1 by LOADING its numbers, never by re-typing them.
    b1 = balanced(store.load(store.BASELINE1))
    b2 = balanced(results)
    df = pd.DataFrame([
        {"condition": c, "baseline1": b1[c], "baseline2_aug": b2[c],
         "aug_gain": b2[c] - b1[c]}
        for c in b2 if c in b1
    ])
    store.save_table("baseline1_vs_baseline2", df)
    print()
    print(df.to_string(index=False))
