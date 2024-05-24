# QFCMolGAN

This code is built on top of the PyTorch implementation for the article [Exploring the Advantages of Quantum Generative Adversarial Networks in Generative Chemistry](https://arxiv.org/abs/2210.16823) available [here](https://github.com/pykao/QuantumMolGAN-PyTorch) (from the main branch).

The primary addition, apart from considerable cleaning-up, is a PyTorch-implementation of a simulation of the QFC system for the input generator. Several input encodings are available, mostly for experimenting.

The quantum circuits are implemented in the `models/q_generator` module, implemented as PyTorch modules so that it works directly with backpropagation for simplicity and simulation speed. In a real experiment, this is of course not possible; the training of the quantum circuit should probably be split into a separate training step (interspersed with training steps of the generator and discriminator).


## Environment

The environment can be installed via conda, mamba, or similar (commands are the same for both conda and mamba):

```bash
mamba env create -f environment.yml
```

You can then activate the environment:

```bash
mamba activate qfcmolgan
```

## Download GDB-9 Dataset

Simply run a bash script in the data directory and the GDB-9 dataset will be downloaded and unzipped automatically together with the required files to compute the NP and SA scores.

```bash
bash data/download_dataset.sh
```

The QM9 dataset is located in the data directory as well.

Feel free to use it.

## Data Preprocessing

Simply run the python script within the data directory.

You need to comment or uncomment some lines of code in the main function.

```bash
python data/sparse_molecular_dataset.py
```

## Running the training script

The training is performed by running the main script

```bash
python main.py
```

You can define the training parameters within the training block of the main function in `main.py`.

## Testing Phase

Simply run the same command to test the system.

You need to comment the training section and uncomment the testing section in the main function of `main.py`, and then run it as before.

## Others

The `results` folder stores the log files, trained models, pretrained quantum circuits, and the testing results.
If `config.use_external_logger` is enabled, the script will try to connect to a wand account and log the training progress there.

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2210.16823,
  doi = {10.48550/ARXIV.2210.16823},

  url = {https://arxiv.org/abs/2210.16823},

  author = {Kao, Po-Yu and Yang, Ya-Chu and Chiang, Wei-Yin and Hsiao, Jen-Yueh and Cao, Yudong and Aliper, Alex and Ren, Feng and Aspuru-Guzik, Alan and Zhavoronkov, Alex and Hsieh, Min-Hsiu and Lin, Yen-Chu},

  keywords = {Quantum Physics (quant-ph), FOS: Physical sciences, FOS: Physical sciences},

  title = {Exploring the Advantages of Quantum Generative Adversarial Networks in Generative Chemistry},

  publisher = {arXiv},

  year = {2022},

  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```

## Credits
This repository was originally imported from:
[pykao/QuantumMolGAN-PyTorch](https://github.com/pykao/QuantumMolGAN-PyTorch) ,
which refers to the following repositories:
 - [nicola-decao/MolGAN](https://github.com/nicola-decao/MolGAN)
 - [ZhenyueQin/Implementation-MolGAN-PyTorch](https://github.com/ZhenyueQin/Implementation-MolGAN-PyTorch)
 - [jundeli/quantum-gan](https://github.com/jundeli/quantum-gan)
