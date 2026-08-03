"""Baseline 1 -- train on clean data only, then measure the silent collapse.

Train a normal pneumonia classifier on clean X-rays, then evaluate it on
corrupted X-rays. The degradation is the problem the project claims to solve,
and this is the floor every later result is measured against.

Once trained, this classifier is FROZEN for the whole project. The recovery
module sits in front of this exact model, so any accuracy recovered later is
attributable to the module rather than to the classifier learning corruptions.

    python scripts/01_baseline1_frozen.py
"""
from _common import config, data, engine, models, parse_args, setup, store

if __name__ == "__main__":
    args = parse_args()
    train, val, test, info, n_classes = setup(args)
    if args.epochs:
        config.CLF_EPOCHS = args.epochs

    clf = models.Classifier(n_classes).to(engine.DEVICE)
    print(f"params: {models.count_params(clf) / 1e6:.2f}M")

    print("\n== training on clean data ==")
    engine.train_classifier(
        clf,
        data.loader(data.Clean(train), shuffle=True),
        data.loader(data.Clean(val)),
        config.BASELINE1_CKPT,
        epochs=config.CLF_EPOCHS,
    )

    print("\n== collapse under corruption ==")
    clf.freeze()
    results = engine.eval_conditions(None, clf, test)

    store.save(store.BASELINE1, results)

    clean_bal = results[data.CLEAN]["balanced_accuracy"]
    print(f"\nclean balanced accuracy: {clean_bal:.4f}")
    collapsed = [c for c, m in results.items() if engine.collapsed(m)]
    if collapsed:
        print("collapsed to a single class on:", ", ".join(collapsed))
    print("\nNote: raw accuracy stays high where balanced accuracy is at 0.50 --"
          "\nthat gap is the 'silent' part of silent collapse.")
