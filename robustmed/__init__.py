"""Lightweight test-time image recovery for robust medical image classification.

A ~1k-parameter autoencoder placed in front of a FROZEN classifier, recovering
accuracy under MedMNIST-C corruptions without retraining the classifier.

Import submodules explicitly:

    from robustmed import config, data, models, engine, figures, store

Nothing is re-exported here on purpose -- so `import robustmed.config` doesn't
drag in matplotlib, sklearn, and the corruption registry.

Experiment stages live in scripts/ and are independent: each writes to
results/ and later stages read from there. See docs/EXPERIMENTS.md.
"""
__version__ = "0.1.0"
