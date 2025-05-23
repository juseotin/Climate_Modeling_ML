\documentclass{article}

% if you need to pass options to natbib, use, e.g.:
%     \PassOptionsToPackage{numbers, compress}{natbib}
% before loading neurips_2024


% ready for submission
%\usepackage{neurips_2024}


% to compile a preprint version, e.g., for submission to arXiv, add add the
% [preprint] option:
     \usepackage[preprint]{neurips_2024}


% to compile a camera-ready version, add the [final] option, e.g.:
%     \usepackage[final]{neurips_2024}


% to avoid loading the natbib package, add option nonatbib:
%    \usepackage[nonatbib]{neurips_2024}


\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{hyperref}       % hyperlinks
\usepackage{url}            % simple URL typesetting
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{xcolor}         % colors
\usepackage{graphicx}
\usepackage{amsmath}


\title{CSE 151B Project Milestone Report}


% The \author macro works with any number of authors. There are two commands
% used to separate the names and addresses of multiple authors: \And and \AND.
%
% Using \And between authors leaves it to LaTeX to determine where to break the
% lines. Using \AND forces a line break at that point. So, if LaTeX puts 3 of 4
% authors names on the first line, and the last on the second line, try using
% \AND instead of \And before the third author name.


\author{%
  Justin M. Seo\thanks{Personal LinkedIn: \url{https://www.linkedin.com/in/justinseodsc}} \\
  Department of Data Science\\
  University of California, San Diego\\
  La Jolla, CA 92092 \\
  \texttt{miseo.ucsd.edu} \\
  % examples of more authors
  % \And
  % Coauthor \\
  % Affiliation \\
  % Address \\
  % \texttt{email} \\
  % \AND
  % Coauthor \\
  % Affiliation \\
  % Address \\
  % \texttt{email} \\
  % \And
  % Coauthor \\
  % Affiliation \\
  % Address \\
  % \texttt{email} \\
  % \And
  % Coauthor \\
  % Affiliation \\
  % Address \\
  % \texttt{email} \\
}


\begin{document}


\maketitle


%\begin{abstract}
  %The abstract paragraph should be indented \nicefrac{1}{2}~inch (3~picas) on
  %both the left- and right-hand margins. Use 10~point type, with a vertical
  %spacing (leading) of 11~points.  The word \textbf{Abstract} must be centered,
  %bold, and in point size 12. Two line spaces precede the abstract. The abstract
  %must be limited to one paragraph.
%\end{abstract}
%

\section{Task Description and Exploratory Analysis}

\subsection{Problem A: Task Description}
The goal of this project is to reproduce a physics-based climate simulation using deep learning. Specifically, I aim to forecast surface air temperature (tas) and precipitation (pr) in different Shared Socioeconomic Pathway (SSP) scenarios by modeling their dependence on variables that are related to climate.

The dataset consists of monthly snapshots of global climate variables structured as spatiotemporal grids. Each sample has a spatial resolution of $48 \times 72$ (latitude $\times$ longitude). The input includes 5 climate forcing variables (CO$_2$, CH$_4$, SO$_2$, BC, and rsdt), while the output consists of 2 target variables: tas and pr.

Mathematically, the input can be represented as $x \in \mathbb{R}^{T \times C \times H \times W}$, where $T$ is the number of time steps (months), $C=5$ is the number of input channels, and $H \times W = 48 \times 72$ denotes the spatial grid. The corresponding output is $y \in \mathbb{R}^{T \times 2 \times H \times W}$.

The learning objective is to train a neural network $f_\theta$ to minimize the mean squared error (MSE) between predicted and true outputs:
\[
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \left\| f_\theta(x_i) - y_i \right\|^2
\]
This loss encourages the model to capture both the spatial patterns and temporal dynamics inherent in the climate data.

\subsection{Problem B: Exploratory Data Analysis}
To understand the dataset’s structure, I visualized the spatial distribution of the target variables \texttt{tas} and \texttt{pr} using the validation set. As shown in Figure~\ref{fig:eda_targets}, surface air temperature (\texttt{tas}) exhibits strong zonal gradients, with higher values over land and greater variability at mid-latitudes. Precipitation (\texttt{pr}) shows the most heterogeneity near the equator, consistent with tropical rainfall belts.

\begin{figure}[h!]
\centering
\includegraphics[width=0.95\linewidth]{eda.png}
\caption{Mean and standard deviation of the target variables \texttt{tas} (surface air temperature) and \texttt{pr} (precipitation) across the validation set. These maps highlight spatial trends and variability.}
\label{fig:eda_targets}
\end{figure}

The dataset contains approximately 2943 training samples and 120 validation samples, each defined on a $48 \times 72$ latitude-longitude grid. Inputs include five forcing variables: CO$_2$, CH$_4$, SO$_2$, BC, and rsdt. The output targets are tas and pr. Variables were standardized using z-score normalization to ensure stability during training.

Exploratory plots revealed clear trends across emission scenarios—higher-emission SSPs are associated with elevated surface temperatures. Seasonal and latitudinal patterns are also visible in both inputs and outputs.

\begin{figure}[h!]
\centering
\includegraphics[width=0.9\linewidth]{zonal_means.png}
\caption{Zonal (latitude-wise) means of surface temperature (tas) and precipitation (pr) across the validation set. This highlights latitudinal climate structure, such as tropical rainfall and polar temperature gradients.}
\label{fig:zonal_means}
\end{figure}

As a deeper diagnostic, I computed the zonal mean (i.e., longitudinal average) for tas and pr across all validation months. Figure~\ref{fig:zonal_means} shows that \texttt{tas} increases from equator to poles, while \texttt{pr} peaks near the equator—patterns that reflect known phenomena such as polar amplification and equatorial rainfall bands. These insights validate the dataset’s physical realism and underscore the need for models capable of capturing such spatial structures.

\section{Model and Experiment Design}

\subsection{Problem A: Training Pipeline}
I configured the training pipeline using PyTorch Lightning with a modular design and centralized configuration to ensure reproducibility. All training was conducted locally on a CPU-only machine within a Jupyter Notebook environment.

The dataset was loaded from a local Zarr archive located at \texttt{/Users/justinseo/Downloads/processed\_data\_cse151b\_v2\_corrupted\_ssp245.zarr}. The model was trained on five input variables (CO$_2$, CH$_4$, SO$_2$, BC, and rsdt) and two output targets (tas and pr). Climate data from three SSP scenarios (SSP126, SSP370, and SSP585) were used for training, while SSP245 was held out entirely for testing. The final 120 months of SSP370 and SSP245 were reserved for validation and test splits, respectively.

\textbf{Training configuration:}
\begin{itemize}
    \item Optimizer: Adam
    \item Learning rate: $1 \times 10^{-3}$
    \item Batch size: 64
    \item Epochs: 10
    \item Device: Auto-detected (CPU used)
    \item Number of dataloader workers: 0
    \item Precision: 32-bit
    \item Sanity validation steps: 0 (disabled)
\end{itemize}

I also set \texttt{pl.seed\_everything(42)} to fix the random seed and ensure consistent results across runs. Each epoch required approximately one hour to complete, resulting in a total training time of about 10 hours. This was primarily due to hardware limitations, as the training was conducted on a CPU without multiprocessing. In future work, I plan to migrate to GPU-based infrastructure and enable parallel data loading to improve training efficiency.

\subsection{Problem B: Model Description}
I implemented a convolutional architecture named \textbf{SimpleCNN}, designed to model spatial climate patterns over latitude-longitude grids. The network leverages residual connections to facilitate stable training and enable deeper feature extraction.

\textbf{Input/Output:}  
The model receives input tensors of shape $(B, 5, 48, 72)$, where $B$ is the batch size corresponding to individual time steps (months) grouped into batches, $5$ is the number of input variables, and $48 \times 72$ are the spatial grid dimensions. It produces output tensors of shape $(B, 2, 48, 72)$, corresponding to predicted surface air temperature (tas) and precipitation (pr).

\textbf{Architecture Summary:}
\begin{itemize}
    \item \textbf{Initial Block:} 2D convolution (kernel size = 3) $\rightarrow$ Batch Normalization $\rightarrow$ ReLU
    \item \textbf{Residual Stack:} 4 residual blocks, each consisting of two convolutions and a skip connection. Intermediate blocks double the number of channels.
    \item \textbf{Dropout:} Spatial dropout (Dropout2D) with a rate of 0.2, applied after the residual stack
    \item \textbf{Final Block:} 2D convolution $\rightarrow$ Batch Normalization $\rightarrow$ ReLU $\rightarrow$ final Conv2D reducing the channel count to 2
\end{itemize}

\textbf{Residual Block Definition:}
\[
\text{ResidualBlock}(x) = \text{ReLU}(\text{BN}_2(\text{Conv}_2(\text{ReLU}(\text{BN}_1(\text{Conv}_1(x))))) + \text{skip}(x))
\]
A skip connection is applied directly if input and output dimensions match, or includes a $1\times1$ convolution when downsampling is needed.

This model contains approximately \textbf{10.7 million parameters}. It focuses purely on spatial representation learning and serves as a stable and efficient CNN baseline without incorporating temporal modeling.



\section{Results and Future Work}

\subsection{Problem A: Evaluation Metrics}
Validation performance improved consistently over training epochs, as shown in Table~\ref{tab:val_metrics}. Metrics include the root mean squared error (RMSE), time-mean RMSE (computed over the 10-year average), and time-stddev MAE (mean absolute error of the standard deviation over time). These metrics are reported separately for surface temperature (tas) and precipitation (pr).

\begin{table}[h!]
\centering
\begin{tabular}{c|ccc|ccc}
\toprule
\textbf{Epoch} & \textbf{tas RMSE} & \textbf{t-mean} & \textbf{t-stddev} & \textbf{pr RMSE} & \textbf{p-mean} & \textbf{p-stddev} \\
\midrule
0 & 11.85 & 10.33 & 2.63 & 3.53 & 2.15 & 1.89 \\
1 & 10.20 & 8.91 & 2.06 & 3.45 & 2.00 & 1.74 \\
3 & 7.50 & 6.31 & 1.76 & 3.25 & 1.71 & 1.76 \\
5 & 5.70 & 4.23 & 1.54 & 2.90 & 1.12 & 1.50 \\
9 & 4.47 & 3.06 & 1.09 & 2.68 & 0.74 & 1.40 \\
\bottomrule
\end{tabular}
\caption{Validation performance over epochs. Metrics include RMSE, time-mean RMSE, and time-stddev MAE for surface temperature (tas) and precipitation (pr).}
\label{tab:val_metrics}
\end{table}

\begin{figure}[h!]
\centering
\includegraphics[width=0.8\linewidth]{loss_curve.png}
\caption{Validation RMSE for surface temperature (tas) and precipitation (pr) over training epochs.}
\label{fig:loss_curve}
\end{figure}

Figure~\ref{fig:error_maps}, now placed at page 7, presents the spatial absolute error maps for three validation samples with the highest total error. These maps highlight the geographic regions where the model predictions deviated most from the ground truth.

\textbf{Note on Submission Format:}  
Due to computational limitations, I was only able to generate predictions for the final 120 months (rather than the expected 240 or 360 months). Each training epoch required approximately one hour on a local CPU-based Jupyter Notebook environment, making full-sequence inference infeasible. I attempted to create a padded CSV submission with placeholder values to validate the format on Kaggle, but this approach failed to meet the required submission criteria. As a result, I am currently unable to submit a valid prediction file to the competition platform. For the final report, I plan to re-train and evaluate the model using GPU acceleration and the full test sequence.


\subsection{Problem B: Reflection and Next Steps}
I explored multiple architectural and training design choices, starting from a basic convolutional network and refining it into a residual CNN with batch normalization and dropout. The final model, SimpleCNN, significantly outperformed the baseline in all evaluation metrics.

\begin{table}[h!]
\centering
\begin{tabular}{l|cc|cc}
\toprule
Model & tas RMSE & pr RMSE & Time-Mean RMSE & Time-Stddev MAE \\
\midrule
Baseline ConvNet & 9.78 & 3.82 & 7.25 (tas) & 2.60 (pr) \\
SimpleCNN (final) & 4.47 & 2.68 & 3.06 (tas) & 1.40 (pr) \\
\bottomrule
\end{tabular}
\caption{Comparison of model iterations and validation scores.}
\label{tab:model_comparison}
\end{table}

\textbf{What worked well:}
\begin{itemize}
    \item Incorporating residual connections improved gradient flow and model stability during training.
    \item Batch normalization helped accelerate convergence and enhance generalization.
    \item Spatial dropout effectively reduced overfitting by introducing regularization at the feature map level.
\end{itemize}

\textbf{What didn’t work:}
\begin{itemize}
    \item I attempted to design an entirely new architecture from scratch, independent of the SimpleCNN baseline. However, this model failed to converge due to architectural instability and limited debugging time, and thus could not be evaluated meaningfully.
    \item Predicting precipitation (\texttt{pr}) remained particularly challenging, likely due to its inherently higher spatial and temporal variability compared to temperature.
    \item Hardware constraints (CPU-only training) prevented scaling to deeper architectures and longer input sequences, such as extending predictions beyond 120 months.
\end{itemize}

\textbf{Next steps:}
\begin{itemize}
    \item Integrate temporal modeling using ConvLSTM or 3D convolutional layers to capture dynamic dependencies.
    \item Explore transformer-based encoders for spatial-temporal attention mechanisms.
    \item Explicitly encode SSP scenario identifiers using one-hot vectors or learned embeddings.
    \item Incorporate physics-informed loss functions to improve realism and physical consistency of predictions.
    \item Migrate training to GPU environments and enable parallelized data loading to reduce training time and enable experimentation with larger models.
\end{itemize}



\section*{GitHub Repository}
\url{https://github.com/juseotin/Climate_Modeling_ML}

\section*{References}
{
\small

[1] Pincus, R., Forster, P. M., & Stevens, B. (2015). The Radiative Forcing Model Intercomparison Project (RFMIP): experimental protocol for CMIP6. *Geoscientific Model Development*, 8(1), 1–11. \url{https://doi.org/10.5194/gmd-8-1-2015}

[2] Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In *Advances in Neural Information Processing Systems*, 32. \url{https://papers.nips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html}

[3] Falcon, W., & The PyTorch Lightning team. (2019). PyTorch Lightning. \url{https://github.com/Lightning-AI/lightning}

[4] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (CVPR), 770–778. \url{https://doi.org/10.1109/CVPR.2016.90}
Used for understanding residual learning.

[5] Zarr Developers. Zarr v2 Specification. \url{https://zarr.readthedocs.io}

[6] Dask Development Team. (2016). Dask: Library for dynamic task scheduling. \url{https://www.dask.org}

[7] Xarray Developers. (2022). Xarray: N-D labeled arrays and datasets in Python. \url{http://xarray.dev}

[8] Kaggle. CSE151B Spring 2025 Competition. \url{https://www.kaggle.com/competitions/cse151b-spring2025-competition}

[9] OpenAI. (2024). ChatGPT (Mar 14 version) [Large language model]. \url{https://chat.openai.com}  
Used for code assistance, debugging, and writing support (Grammar Fix) during the CSE151B project milestone.
}

\section*{Acknowledgments}
Thanks to the CSE151B TAs and instructors for guidance. Repository access granted to: salv47, nishant42491, atong28, AZA-2003, VedantMohann, charliespy, DivyamSengar.

\begin{figure}[h!]
\centering
\includegraphics[width=\linewidth]{high_error_maps.png}
\caption{Spatial absolute error maps for tas and pr on validation samples with highest error. Brighter regions indicate larger prediction errors.}
\label{fig:error_maps}
\end{figure}

\end{document}
