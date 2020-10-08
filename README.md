<p align="center">
  <img src="https://raw.githubusercontent.com/Unbabel/COMET/master/docs/source/_static/img/COMET_lockup-dark.png">
  <br />
  <br />
  <a href="https://github.com/Unbabel/COMET/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Unbabel/COMET" /></a>
  <a href="https://github.com/Unbabel/COMET/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Unbabel/COMET" /></a>
  <a href=""><img alt="PyPI" src="https://img.shields.io/pypi/v/unbabel-comet" /></a>
  <a href="https://github.com/psf/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-black" /></a>
</p>

**Note:** This is a Pre-Release Version. We are currently working on results for the WMT2020 shared task and will likely update the repository in the beginning of October (after the shared task results).

## Quick Installation

We recommend python 3.6 to run COMET.

Detailed usage examples and instructions can be found in the [Full Documentation](https://unbabel.github.io/COMET/html/index.html).

Simple installation from PyPI

```bash
pip install unbabel-comet
```

To develop locally:
```bash
git clone https://github.com/Unbabel/COMET
pip install -r requirements.txt
pip install -e .
```

## Scoring MT outputs:

### Via Bash:

Example:

```bash
echo -e "Hello world\nThis is a sample" >> src.en
echo -e "Oi mundo\neste é um exemplo" >> hyp.pt
echo -e "Olá mundo\nisto é um exemplo" >> ref.pt
```

```bash
comet score -s src.en -h hyp.pt -r ref.pt
```

You can export your results to a JSON file using the `--to_json` flag and select another model/metric with `--model`.

```bash
comet score -s src.en -h hyp.pt -r ref.pt --model wmt-large-hter-estimator --to_json segments.json
```

### Via Python:

```python
from comet.models import download_model
model = download_model("wmt-large-da-estimator-1719", "path/where/to/save/models/")
data = [
    {
        "src": "Hello world!",
        "mt": "Oi mundo!",
        "ref": "Olá mundo!"
    },
    {
        "src": "This is a sample",
        "mt": "este é um exemplo",
        "ref": "isto é um exemplo!"
    }
]
model.predict(data)
```

### Simple Pythonic way to convert list or segments to model inputs:

```python
source = ["Hello world!", "This is a sample"]
hypothesis = ["Oi mundo!", "este é um exemplo"]
reference = ["Olá mundo!", "isto é um exemplo!"]

data = {"src": source, "mt": hypothesis, "ref": reference}
data = [dict(zip(data, t)) for t in zip(*data.values())]

model.predict(data)
```


## Model Zoo:

| Model              |               Description                        |
| :--------------------- | :------------------------------------------------ |
| ↑`wmt-large-da-estimator-1719` | **RECOMMENDED:** Estimator model build on top of XLM-R (large) trained on DA from WMT17, WMT18 and WMT19 |
| ↑`wmt-base-da-estimator-1719` | Estimator model build on top of XLM-R (base) trained on DA from WMT17, WMT18 and WMT19 |
| ↓`wmt-large-hter-estimator` | Estimator model build on top of XLM-R (large) trained to regress on HTER. |
| ↓`wmt-base-hter-estimator` | Estimator model build on top of XLM-R (base) trained to regress on HTER. |
| ↑`emnlp-base-da-ranker`    | Translation ranking model that uses XLM-R to encode sentences. This model was trained with WMT17 and WMT18 Direct Assessments Relative Ranks (DARR). |

#### QE-as-a-metric:

| Model              |               Description                        |
| -------------------- | -------------------------------- |
| `wmt-large-qe-estimator-1719` | Quality Estimator model build on top of XLM-R (large) trained on DA from WMT17, WMT18 and WMT19. |

## Train your own Metric: 

Instead of using pretrained models your can train your own model with the following command:
```bash
comet train -f {config_file_path}.yaml
```

### Supported encoders:
- [Learning Joint Multilingual Sentence Representations with Neural Machine Translation](https://arxiv.org/abs/1704.04154)
- [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/abs/1812.10464)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [XLM-R: Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/pdf/1911.02116.pdf)


### Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="experiments/lightning_logs/"
```

## Download Command: 

To download public available corpora to train your new models you can use the `download` command. For example to download the APEQUEST HTER corpus just run the following command:

```bash
comet download -d apequest --saving_path data/
```

## unittest:
```bash
pip install coverage
```

In order to run the toolkit tests you must run the following command:

```bash
coverage run --source=comet -m unittest discover
coverage report -m
```
