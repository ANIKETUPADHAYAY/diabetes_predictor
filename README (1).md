# Plant Disease Classification using Traditional Machine Learning

This project aims to classify plant diseases using handcrafted image features and traditional machine learning algorithms (no deep learning or transfer learning involved). It is built specifically for internship tasks or academic assignments with limitations on using CNNs or pretrained models.

## 📁 Dataset

The dataset should be organized in the following structure:

```
dataset_path_here/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Apple___Black_rot/
│   ├── image1.jpg
│   └── ...
...
```

You can use the [Plant Pathology 2020 dataset from Kaggle](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) or any similar dataset.

## 🧠 Model Workflow

1. Extract color histogram features using OpenCV.
2. Encode labels with `LabelEncoder`.
3. Train a `RandomForestClassifier`.
4. Evaluate the model with a classification report and confusion matrix.

## 🧪 Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

Install requirements via:

```bash
pip install opencv-python numpy scikit-learn matplotlib seaborn
```

## 🚀 Run Instructions

1. Place your dataset in a local folder.
2. Replace the value of `data_dir` in the notebook with the correct dataset path.
3. Run all cells in the notebook.

## 📊 Output

- Classification report with precision, recall, F1-score
- Confusion matrix heatmap
