
# Climate Modeling with Deep Learning (CSE 151B Project Milestone)

This repository contains the code, data configurations, and results for my project milestone in CSE 151B at UC San Diego. The goal of this project is to emulate a physics-based climate model using deep learning, particularly a residual convolutional neural network (CNN), to predict future surface air temperature (`tas`) and precipitation (`pr`) under various SSP emission scenarios.

## 📁 Project Structure

- `models/` – Contains model architecture (SimpleCNN with residual blocks)
- `scripts/` – Code for training, evaluation, and submission generation
- `data/` – Zarr dataset input (not included in repo; stored locally at runtime)
- `plots/` – Figures generated for EDA and model evaluation (e.g., error maps, loss curves)
- `submission/` – (Optional) CSV predictions if applicable

## 🔍 Project Overview

- **Inputs**: CO₂, CH₄, SO₂, BC, rsdt (5 variables)
- **Outputs**: Surface temperature (`tas`) and precipitation (`pr`)
- **Spatial resolution**: 48×72 grid
- **Temporal**: Monthly snapshots, 120 months validation, 120 months test
- **Training Scenarios**: SSP126, SSP370, SSP585
- **Test Scenario**: SSP245 (held out)

## 🚀 Model Architecture

Implemented a residual convolutional network (`SimpleCNN`) with:
- Initial Conv2D + BatchNorm + ReLU
- 4 Residual Blocks
- Dropout2D
- Final Conv2D to 2 output channels

Model trained using PyTorch Lightning.

## ⚙️ Training Configuration

- Optimizer: Adam
- Learning Rate: 1e-3
- Batch Size: 64
- Epochs: 10
- Precision: 32-bit
- Hardware: CPU only (Jupyter Notebook)

## 📊 Key Results

| Model               | tas RMSE | pr RMSE | Time-Mean RMSE | Time-Stddev MAE |
|--------------------|----------|---------|----------------|-----------------|
| Baseline ConvNet   | 9.78     | 3.82    | 7.25 (tas)     | 2.60 (pr)       |
| SimpleCNN (final)  | 4.47     | 2.68    | 3.06 (tas)     | 1.40 (pr)       |

## 📈 Visualizations

- `eda.png`: Spatial mean/std of tas/pr
- `zonal_means.png`: Latitude trends of tas/pr
- `loss_curve.png`: Validation loss over epochs
- `high_error_maps.png`: Model error maps on validation set

## 🧠 Next Steps

- Add temporal modeling (ConvLSTM / 3D CNN)
- Try Transformer-based spatial-temporal attention
- Use SSP embeddings or one-hot encodings
- Incorporate physics-informed loss functions
- Transition to GPU for faster training

## 🤖 Tools & Libraries

- PyTorch & PyTorch Lightning
- Dask, Xarray, Zarr
- Matplotlib, Numpy, Pandas
- ChatGPT for debugging and grammar review

## 📜 Citation

If referencing this work, please cite the milestone report or the GitHub repository:
[GitHub Repository](https://github.com/juseotin/Climate_Modeling_ML)

## 🙏 Acknowledgments

Special thanks to the CSE151B teaching team for their support and guidance throughout the project.

---
Justin M. Seo  
Department of Data Science, UC San Diego  
[LinkedIn](https://www.linkedin.com/in/justinseodsc)
