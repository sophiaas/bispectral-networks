# Bispectral Neural Networks

This repository is the official implementation of Bispectral Neural Networks.

## Installation

To install the requirements and package, run:

```
pip install -r requirements.txt
python setup.py install
```

## Datasets

To download the datasets, run:

```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10w3fKdO0eWEe2KxZxpf8YFndXdCNNR8b' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10w3fKdO0eWEe2KxZxpf8YFndXdCNNR8b" -O datasets.zip 
rm -rf /tmp/cookies.txt
unzip datasets.zip
rm -r datasets.zip
```


If your machine doesn't have wget, follow these steps: 
1. Download the zip file [here](https://drive.google.com/file/d/10w3fKdO0eWEe2KxZxpf8YFndXdCNNR8b/view?usp=sharing).
2. Place the file in the top node of this directory, i.e. in `bispectral-networks/`.
3. Run:
    ```
    unzip datasets.zip
    rm -r datasets.zip
    ```

## Training

To train the models in the paper, run the following commands.

```
python train.py --config rotation_experiment
python train.py --config translation_experiment
```

To run on GPU, add the following argument, with the integer specifying the device number, i.e.:


```
--device 0
```

The full set of hyperparameters and training configurations are specified in the config files in the ```configs/``` folder.

To view learning curves in Tensorboard, run:
```
tensorboard --logdir logs/
```

## Pre-trained Models

The pre-trained models are included in the repo, in the following locations:

```
logs/rotation_model/
logs/translation_model/
```


## Results and Figures

All results and figures from the paper are generated in the Jupyter notebooks located at:

```
notebooks/rotation_experiment_analysis.ipynb
notebooks/translation_experiment_analysis.ipynb
```

## License

This repository is licensed under the MIT License.  

