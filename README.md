# Combined Cycle Power Plant - Energy Output Prediction (PyTorch)

A PyTorch-based regression model using an Artificial Neural Network (ANN) to predict the net hourly electrical energy output (PE) of a Combined Cycle Power Plant. The predictions are based on four ambient environmental features: Temperature, Vacuum, Pressure, and Humidity.

---

## Dataset & Features

The dataset (`powerplant_data.csv`) contains 9,568 hourly records collected from a power plant operating under full load. 

The model uses the following input variables:
* **AT**: Ambient Temperature ($1.81^\circ\text{C}$ to $37.11^\circ\text{C}$)
* **V**: Exhaust Vacuum ($25.36\text{ cm Hg}$ to $81.56\text{ cm Hg}$)
* **AP**: Ambient Pressure ($992.89\text{ mbar}$ to $1033.30\text{ mbar}$)
* **RH**: Relative Humidity ($25.56\%$ to $100.16\%$)

**Target variable:**
* **PE**: Net hourly electrical energy output ($420.26\text{ MW}$ to $495.76\text{ MW}$)

---

## Model Architecture

The neural network is built with PyTorch and has the following layer structure:
* **Input Layer**: 4 inputs (corresponding to the scaled features)
* **Hidden Layer 1**: 6 neurons with ReLU activation
* **Hidden Layer 2**: 6 neurons with ReLU activation
* **Output Layer**: 1 neuron (linear prediction of energy output)

### Network Structure
![Model Architecture](network_diagram.png)

```
Input (4 features) ──> Dense (6) ──> ReLU ──> Dense (6) ──> ReLU ──> Output (1)
```

---

## Training Setup

* **Preprocessing**: The features are split into training (80%) and validation (20%) sets, then normalized using Scikit-Learn's `StandardScaler`.
* **Optimization**: The model is trained using **Adam** optimizer and **MSE Loss** (`nn.MSELoss`) over 100 epochs with a batch size of 32.
* **Checkpointing**: The training script monitors validation loss and saves the best model state dict to `best_model.pt`.
* **Analysis**: Epoch loss curves (training vs validation) are plotted to monitor convergence and check for overfitting.

---

## Repository Structure

* `ANN_Regression.ipynb` - Jupyter notebook containing data loading, scaling, model training, evaluation, and plotting.
* `powerplant_data.csv` - The raw powerplant dataset.
* `best_model.pt` - The saved PyTorch model weights from the best training epoch.
* `network_diagram.png` - Detailed visualization of the neural network (referenced in this README).
* `model_architecture.png` - Standard block diagram representation of the network.
* `.gitignore` - Standard Python and Jupyter notebook temporary file configurations.

---

## Setup and Running

To run this notebook, you will need PyTorch, Scikit-Learn, Pandas, Numpy, and Matplotlib.

1. **Install requirements:**
   ```bash
   pip install torch pandas numpy scikit-learn matplotlib jupyter
   ```

2. **Launch the notebook:**
   ```bash
   jupyter notebook ANN_Regression.ipynb
   ```
