![](https://img.shields.io/badge/pymc-3.9.3-green) ![](https://img.shields.io/badge/tensorflow_probability-0.11.0-yellow) ![](https://img.shields.io/badge/license-GNU%20GPL%20v3.0-blue)

# Practical Bayes



This repository reflects my current state of knowledge of Bayesian Inference.
It contains notes, notebooks, and tools from my talks, workshops, and readings since January 2018.

Are you interested in learning more about the following topics?

- Probabilistic programming
- Bayesian Data Analysis
- How to think like a Bayesian
- Applications of Bayesian Inference

So am I. Follow this repository for more updates!

## Table of Contents

|      | Title                                                        | Description / Features                                       | Resources                                                    |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | Bayesian Thinking with Tensorflow Probability                | [Talk at TFUG Chennai Online Meetup](https://twitter.com/TFUGChennai/status/1295763421149855744?s=20) | [Notebooks](tfp/)                                            |
|      |                                                              |                                                              | [Slides](https://drive.google.com/file/d/1I4BHlQZBo49pGy77LH3Xc_b2Hg41k7e7/view?usp=sharing) |
| 2    | `tfp_helper`: Helper Library for prototyping with Tensorflow Probability | Inference Button                                             | [Code](https://github.com/suriyadeepan/tfp_helper)           |
|      |                                                              | `arviz` adapter for Plotting                                 |                                                              |
|      |                                                              | Smart Progress Bar Bar ▓▓▓▓▓▓▓▓▓▓░░░░░ 67%                   |                                                              |



## Getting Started

Instructions on getting the repo setup and running on your local machine.

### Prerequisites

Install requirements for `tensorflow_probability`-based notebooks.

```bash
pip install -r tfp/requirements.txt
```

Install requirements for `pymc3`-based notebooks.

```bash
pip install -r pymc/requirements.txt
```

Install global requirements for running all the notebook.

```bash
pip install -r requirements_global.txt
```

## Running the tests

Instructions for running tfp tests.

```bash
pytest tfp/tests.py
```

Running pymc tests.

```bash
pytest pymc/tests.py
```

## Acknowledgements

- Examples from Cameron Davidson-Pilon's Book [Bayesian Methods for Hackers](https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/)
- Data borrowed from Robert R.F. DeFilippi's blog on [Bayesian Analysis of Traffic Patterns](https://medium.com/@rrfd/bayesian-analysis-for-traffic-patterns-480e71a680ab)
- Covid-19 Bayesian Forecast plot borrowed from [Inferring change points in the spread of COVID-19 reveals the effectiveness of interventions](https://science.sciencemag.org/content/369/6500/eabb9789)
- Template for slides borrowed from Junpen Lao's [A Hitchhiker's Guide to designing a Bayesian library in Python](https://docs.google.com/presentation/d/1xgNRJDwkWjTHOYMj5aGefwWiV8x-Tz55GfkBksZsN3g/edit?usp=sharing)

## License

This project is licensed under the GNU GPL v3.0 License - see the [LICENSE](LICENSE) file for details.
