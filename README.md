Cost-Sensitive Focal Label Smoothing (CP-FLS)

A novel loss function for imbalanced classification that combines Cost-Sensitive Weighting, Focal Loss, and Label Smoothing into one unified framework.

📌 Overview

In real-world scenarios like fraud detection, medical diagnosis, and intrusion detection systems, datasets are often highly imbalanced.

Cross-Entropy Loss → biased towards majority classes.

Focal Loss → improves recall but hurts calibration.

Label Smoothing → improves calibration but weakens recall.

CP-FLS addresses all three challenges simultaneously:

Class imbalance

Hard example focusing

Prediction calibration

🧪 Key Features

Novel loss function (CP-FLS) combining imbalance handling and calibration.

Implemented in PyTorch from scratch.

Works on imbalanced CIFAR-10 dataset.

Evaluation with Accuracy, F1, AUC, PR-AUC, Calibration Error (ECE), and Brier Score.

Includes training scripts, utilities, and visualization tools.

⚙️ Installation

Clone the repo and install requirements:

git clone https://github.com/your-username/cpfls-loss.git
cd cpfls-loss
pip install -r requirements.txt

📂 Project Structure
├── cpfls_loss.py   # Implementation of CE, Focal, LS, CP-FLS
├── model.py        # SimpleCNN model architecture
├── train.py        # Training loop and evaluation
├── utils.py        # Data preprocessing, metrics, visualization
├── best_model.pth  # Trained CP-FLS model weights
├── README.md       # Project documentation

▶️ Usage
Training
python train.py --loss cpfls --epochs 50 --batch_size 128

Evaluate
python train.py --evaluate --model best_model.pth

Available Loss Options

ce → Cross-Entropy

focal → Focal Loss

ls → Label Smoothing

cpfls → Cost-Sensitive Focal Label Smoothing

📊 Results (Sample)
Loss	Accuracy	F1	AUC	PR-AUC	ECE ↓	Brier ↓
CE	0.78	0.52	0.83	0.41	0.092	0.186
Focal	0.75	0.58	0.86	0.47	0.104	0.174
LS	0.77	0.55	0.84	0.44	0.071	0.165
CP-FLS	0.79	0.63	0.89	0.54	0.059	0.151
📈 Visualizations

Training curves (Loss, Accuracy, F1, AUC)

Precision-Recall Curves

Calibration Diagrams

Confusion Matrices

(See /outputs folder or generated plots during training.)

🧑‍💻 Author

Venkat Ashwin Kumar

📧 Email: ashwinkumar092005@gmail.com



📖 References

This work is based on insights from 25+ SCI/Scopus indexed journal papers (with DOIs), covering loss functions, calibration, and imbalanced learning.

📌 License

This project is released under the MIT License.
