# Implementation of Stone 2019's paper: Calibrating Rough Volatility Models using a Convolutional Neural Network Approach
This repository contains a complete notebook along with pre-trained CNN models for calibrating rough volatility models using a one-dimensional convolutional neural network (1D CNN).

---

## Description

In this project, we use a CNN-based approach to predict the Hölder exponent (H) of a stochastic process. The notebook details:

- **Simulation and Data Preparation:**  
  - Generation of trajectories using the rough Bergomi model and fractional Brownian motion (fBm) processes.  
  - Functions for computing lagged differences and performing least squares regression.

- **CNN Architecture and Training:**  
  - A 1D CNN architecture including convolutional layers, LeakyReLU activation, max pooling, dropout, and dense layers.  
  - Network parameters and training procedures using early stopping.

- **Evaluation and Benchmarking:**  
  - Comparison of the CNN’s performance against a least squares regression approach.  
  - Visualization of learning curves and robustness tests on various processes (e.g., Ornstein-Uhlenbeck, variations of parameters η and H).

- **Calibration on Real Data:**  
  - Application of the trained CNN models to historical realized volatility data (sourced from [rv_dataset.csv](https://github.com/andymogul/SpilloverVolPrediction)) to predict the Hölder exponent.

**Institution:** Machine Learning in Finance (ENSAE Paris - IP Paris)  
**Inspired by:** The approach proposed by Henry Stone

---

## Repository Structure

- **Main Notebook:**  
  `notebook.ipynb`  
  Contains detailed explanations, code, and visualizations for simulation, training, and calibration of the models.

- **Pre-trained CNN Models:**  
  - `cnn_model_discretized.h5` and `cnn_model_discretized.keras`  
    Models trained with H values generated using the discretized method.
  - `cnn_model_uniform.h5` and `cnn_model_uniform.keras`  
    Models trained with H values sampled from a uniform distribution.
  - `cnn_model_beta.h5` and `cnn_model_beta.keras`  
    Models trained with H values sampled from a Beta(1,9) distribution.

- **Data:**  
  `rv_dataset.csv`  
  Realized volatility dataset used for the final calibration.

---

## Dependencies

The main libraries used in this project are:

- Python 3.x
- NumPy
- Pandas
- SciPy
- Matplotlib
- scikit-learn
- TensorFlow & Keras
- fbm (for simulating fractional Brownian motion)

To install the dependencies, run for example:

```bash
pip install numpy pandas scipy matplotlib scikit-learn tensorflow keras fbm
```

## How to Copy and Cite

### Copying
You are free to copy or fork this repository for your own research and development purposes. To clone the repository, use the following command:

```bash
git clone https://github.com/adeladrai/calibrating_rough_volatility_with_cnn/
```
### Citation 

If you use this repository or any part of its code in your research or publications, please cite it as follows:


```bibtex
@misc{adrai2024calibrating,
    title={Implementation of Stone 2019 paper: Calibrating Rough Volatility Models using a Convolutional Neural Network Approach},
    author={Adel A.},
    year={2024},
    note={Available at \url{https://github.com/adeladrai/calibrating_rough_volatility_with_cnn/}},
}
```

