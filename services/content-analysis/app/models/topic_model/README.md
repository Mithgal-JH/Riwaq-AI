# BinX Transformer Topic Classifier

Selected system: microsoft__deberta_v3_base

Validation Micro-F1: 0.8714
Test Micro-F1: 0.8734
Test Macro-F1: 0.7902

## Inference

```python
from inference import BinXClassifier
classifier = BinXClassifier('.')
print(classifier.predict('Build an ESP32 robot with computer vision.'))
```

The classifier returns every topic above its calibrated threshold, sorted by confidence, with a maximum of 6. If none pass, it returns `Unclassified`.

Review `dataset_license_report.json` before production deployment.
