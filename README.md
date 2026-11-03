# CIFAR-10 CNN in PyTorch

A small convolutional network trained on CIFAR-10 (60,000 32×32 colour images, 10 classes), packaged as a reproducible project: config-driven training, a held-out validation split, a single final test evaluation, and error analysis.


## Results

| Params  | Augmentation | Val acc | Test acc |
|---------|---|---|---|
| 357,034 | crop + flip | 75.94% | 75.29% |

Outputs written to `results/` after a run: `training_curves.png`, `confusion_matrix.png`, `per_class_accuracy.png`, `misclassified.png`, `metrics.json`.


### Error analysis
_After evaluating, note the top confusions from `results/metrics.json` (typically cat↔dog and car↔truck for small CNNs on CIFAR-10) and what you see in `misclassified.png`._


## Project layout

```
configs/config.ini    hyperparameters and paths
src/data.py           loaders, augmentation, train/val/test split
src/model.py          CNN definition
src/train.py          training loop (CLI)
src/evaluate.py       test metrics + plots
src/predict.py        single-image inference
tests/test_model.py   shape, gradient and regression tests
notebooks/            original Colab notebook (reference)
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.train
python -m src.evaluate --checkpoint models/best.pt --history results/run_history.json
python -m src.predict some_image.jpg --checkpoint models/best.pt
python -m pytest -q
```

On Colab, choose a GPU runtime (Runtime → Change runtime type), then:
As colab runs on virtual machines no need to create virtual environment.

```python
!git clone https://github.com/AMaddodi/CNN_Project.git
%cd CNN_Project/
%pip install -r requirements.txt
%python -m src.train
%python -m src.evaluate --checkpoint models/best.pt --history results/run_history.json
%python -m src.predict some_image.jpg --checkpoint models/best.pt
%python -m pytest -q

```

## Limitations and next steps

- Small CNN; a ResNet-style network or pretrained ResNet18 would score noticeably higher.
- Single seed; report mean ± std over several seeds for a fair ablation.
- 32×32 inputs limit what any model can distinguish for visually similar classes.
- Ideas: mixup/cutmix, label smoothing, experiment tracking (MLflow/W&B), a small Gradio demo.
