# Combined Cycle Power Plant Energy Output Prediction

This repository contains a Deep Learning project that implements an **Artificial Neural Network (ANN)** in PyTorch to predict the net hourly electrical energy output ($PE$) of a Combined Cycle Power Plant based on ambient environmental readings.

---

## 📊 Dataset Description

The dataset (`powerplant_data.csv`) contains 9,568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011), when the power plant was set to work with a full load.

### Features
1. **AT (Ambient Temperature)**: Range: $1.81^\circ\text{C}$ to $37.11^\circ\text{C}$.
2. **V (Exhaust Vacuum)**: Range: $25.36\text{ cm Hg}$ to $81.56\text{ cm Hg}$.
3. **AP (Ambient Pressure)**: Range: $992.89\text{ millibar}$ to $1033.30\text{ millibar}$.
4. **RH (Relative Humidity)**: Range: $25.56\%$ to $100.16\%$.

### Target
* **PE (Net hourly electrical energy output)**: $420.26\text{ MW}$ to $495.76\text{ MW}$.

---

## 🧠 Model Architecture

The Artificial Neural Network is built using **PyTorch** and consists of:
* **Input Layer**: 4 features (`AT`, `V`, `AP`, `RH`) after Standard Scaling.
* **Hidden Layer 1**: Fully connected (Dense) layer with 6 neurons and a ReLU activation function.
* **Hidden Layer 2**: Fully connected (Dense) layer with 6 neurons and a ReLU activation function.
* **Output Layer**: 1 neuron to predict the continuous target variable (`PE`).

```
Input (4 features) ──> Dense (6) ──> ReLU ──> Dense (6) ──> ReLU ──> Output (1)
```

---

## ⚙️ Training Workflow

The PyTorch training workflow is structured as follows:
1. **Data Preprocessing**: Splits the dataset into train and test sets (80/20 split) and scales the features using `StandardScaler`.
2. **Tensors & DataLoaders**: Converts Numpy arrays to PyTorch Tensors and wraps them in `TensorDataset` and `DataLoader` for batching (Batch size: 32).
3. **Loss Function & Optimizer**: 
   * **Criterion**: Mean Squared Error Loss (`nn.MSELoss()`)
   * **Optimizer**: Adam Optimizer (`optim.Adam`)
4. **Training Process**: Runs for 100 epochs, tracking validation losses, and saving the state dict of the best model (lowest validation loss) to `best_model.pt`.
5. **Loss Visualization**: Plots training vs. validation loss across all epochs.

---

## 📁 Repository Structure

* `ANN_Regression.ipynb`: Jupyter notebook containing the full workflow (loading, scaling, training, evaluating, and visualizing).
* `powerplant_data.csv`: The Combined Cycle Power Plant dataset.
* `best_model.pt`: Saved weights of the trained PyTorch ANN model.
* `.gitignore`: Configured to ignore python/jupyter checkpoints and cache files.
* `README.md`: Project documentation (this file).

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following libraries installed:
```bash
pip install pandas numpy scikit-learn torch matplotlib jupyter
```

### Running the Project

1. Clone the repository to your local machine (once uploaded to GitHub).
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook ANN_Regression.ipynb
   ```
3. Run all cells to preprocess the data, train the ANN, and view the loss curves.
