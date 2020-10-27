# Tensorflow Tutorial Python

https://www.tensorflow.org/

## Setup

The setup is about the software I use at the moment (27.10.2020).

### Python

Python 3.8.5 & PIP 20.0.2

sudo apt install python3 python3-pip

For Windows: https://docs.python.org/3/using/windows.html

### Python Environment

For this tutorial we use 'virtualenv'.

https://docs.python.org/3/library/venv.html

(It's also possible to use the environment of your choice, e.g. 'Anaconda'.)

python3 -m venv ./venv_tf_tutorial

If you are using Visual Studio Code, it is recommended to select your venv as the Python Interpreter. Once selected, a new opened terminal will automatically activate your virtual environment.

### Tensorflow 2

https://www.tensorflow.org/install

python3 -m pip install tensorflow

### Other Python Libraries

python3 -m pip install matplotlib ipython

#### GPU Support

https://www.tensorflow.org/install/gpu

With a CUDA-enabled graphics card you can profit from GPU acceleration.

Follow the installation instructions for Linux / Windows:

https://www.tensorflow.org/install/gpu#software_requirements

Issues with some cards (e.g. GTX 1660 SUPER):

export TF_FORCE_GPU_ALLOW_GROWTH=true

#### TPU Support

TODO


