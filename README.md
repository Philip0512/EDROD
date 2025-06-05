# EDROD: Entropy Density Ratio Outlier Detection

This repo contains the official implementation for the paper ["Robust Outlier Detection Method Based on Local Entropy and Global Density"](https://www.sciencedirect.com/science/article/pii/S0957417425020433) by Kaituo Zhang, Wei Huang.
![flowchart](/EDROD/utilis/flowchart.png)

**EDROD** is a parameter-free, unsupervised anomaly detection algorithm based on entropy density ratios. It detects suspicious samples in high-dimensional datasets without requiring model training.

## 🚀  Features

- Effective detection of both point anomalies and cluster anomalies.
- Strong robustness to the number of selected nearest neighbors.
- Supports AUC evaluation when ground truth is available
- Clean and modular Python implementation

## 🧪 Experiments

We conducted experiments on real-world datasets and obtain the results.

![flowchart](/EDROD/utilis/real-world-result.png)

## 📦 Installation

We provide the code of EDROD to detect anomaly.

### 🧩 Environment

```bash
cd EDROD
```

Using `pip`:

```bash
pip install -r requirements.txt
```

Or using `conda`:

```bash
conda create -n edrod_env python=3.9 pandas=1.5.3 numpy=1.23.5 scikit-learn=1.2.2
```

### 💻  Usage

```python
python3 main.py --data_name musk --n_neighbors 50
```

If you have a new dataset, you only need to upload it to the dataset folder and specify the corresponding `data_name`. The results can then be obtained directly.

#### Arguments

| Argument        | Description                                 |
| --------------- | ------------------------------------------- |
| `--data_name`   | The name of dataset                         |
| `--n_neighbors` | Number of neighbors for entropy calculation |
| `--path_prefix` | Optional path prefix to dataset files       |

## 📈 Example Output

If you run the example:

```
python3 main.py --data_name musk --n_neighbors 50
```

You will get:

```makefile
AUC: 1.00
```

If you run another example:

```
python3 main.py --data_name waveform --n_neighbors 50
```

You will get:

```makefile
AUC:  0.861487
```

## 📁 File Structure

```bash
.
├── README.md                  # Project documentation (top-level)
├── EDROD/
│   ├── main.py                # Entry point script
│   ├── LICENSE
│   ├── model.py               # EDROD algorithm class
│   ├── README.md              # Internal module documentation
│   ├── requirements.txt       # Python dependencies
│   ├── dataset/               # Folder for datasets
│   └── utils/                 # Utility picture

```

## 📖 Citation

If you use this code in your research, please cite:

```nginx
@article{ZHANG2025128424,
title = {Robust Outlier Detection Method Based on Local Entropy and Global Density},
journal = {Expert Systems with Applications},
pages = {128424},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.128424},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425020433},
author = {Kaituo Zhang and Bingyang Zhang and Wei Huang and Hua Gao and Ning Xu and Rongchun Wan},
keywords = {Outlier detection, Shannon Entropy, KNN algorithm, Kernel Density Estimation},
abstract = {Most outlier-detection algorithms struggle to accurately identify both point anomalies and cluster anomalies simultaneously. Additionally, while K-nearest-neighbor-based methods often perform well across various datasets, their sensitivity to the choice of K remains a significant limitation. To address these challenges, we propose the Entropy Density Ratio Outlier Detection (EDROD) method, introducing a novel Kernel Density Estimation with Sample-Dependent Bandwidth (KDE-SDB) strategy to compute the global density of each sample as a global feature. We also calculate the local Shannon entropy around each sample as a local feature and define the ratio between entropy and global density as a comprehensive abnormality indicator. Experimental results on synthetic and real-world datasets demonstrate that EDROD effectively detects both point and cluster anomalies with high accuracy, while showing strong robustness to the selection of neighboring samples. This makes EDROD highly applicable to diverse real-world scenarios. Our code is available on our GitHub repository11Code:https://github.com/Philip0512/EDROD.}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](/EDROD/LICENSE.txt) file for details.
