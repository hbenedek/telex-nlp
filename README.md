# 📜 telex-nlp
In this project, I attempt to build a small language model, trained on all the articles of the Hungarian news portal [telex.hu](https://telex.hu/), using a character-based tokenizer.


## 🔧 Set up environment
The python environment is managed with [pipenv](https://pipenv.pypa.io/en/latest/install). You can set up your environment with the following steps:

- Run `pipenv lock`to generate the `Pipfile.lock` which lists the version of your python packages.
- Run `pipenv install --dev` to actually create a virtual environment and install the python packages. The flag `--dev` allows to install the development packages (for linting, ...).
- Run `pipenv shell` to activate the virtual environment

## 🚀 Run the DVC pipeline

The ML pipeline is managed with [DVC](https://dvc.org/), here are a few tips on how to use it:

- Run the complete pipeline: `dvc repro`
- Run a specific step of the pipeline with all its dependencies: `dvc repro <step_name>`

DVC Sages:
- scrape        : using the telex api downloads and saves all articles published since 2020 october
- prerpocess    : removes html, tags, and collects all article contents in a single json
- train         : Dataloader and LM model is initialized, training on characterwise in semi-supervised fashion
- evaluate      : calculates corpus perplexity on a test set, generates random text from input context

## 🏗️ Structure
<pre>
.
├── Pipfile                 <- requirements for running the project
├── Pipfile.lock            <- versions of the required packages
├── README.md
├── dvc.lock                <- automatically records the states of the DVC pipeline
├── dvc.yaml                <- lists the stages for the DVC pipeline
├── pyproject.toml          <- contains the build system requirements of the projects
├── notebooks
├── params.py               <- contains the parameters of the project
├── data
│   ├── preprocessed
│   └── raw
└── telex                   <- source code of the project
    ├── models              <- ml model definitions
    │   ├── base_model.py
    │   ├── bigram.py
    │   └── transformer.py
    ├── pipeline            <- scripts for each stage in the DVC pipeline
    │   ├── evaluate
    │   ├── preprocess
    │   ├── scrape          <- scraping articles from telex
    │   └── train           <- model training scripts
    └── utils               <- helper scripts
        ├── dataset.py      <- defines pytorch Dataset object from raw articles
        └── io.py           <- input/output related functions

