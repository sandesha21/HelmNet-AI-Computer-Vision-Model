# HelmNet Helmet Detection

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)
![Project Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg?style=for-the-badge)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Deep%20Learning-orange.svg?style=for-the-badge)
![Safety](https://img.shields.io/badge/Industry-Safety%20Monitoring-red.svg?style=for-the-badge)

---

## 🏷️ Keywords & Topics

**Primary Keywords:** Computer Vision • Deep Learning • Safety Monitoring • Image Classification • Workplace Safety  
**Technical Stack:** TensorFlow/Keras • OpenCV • CNN • Transfer Learning • VGG-16 • Python • NumPy  
**Business Focus:** Safety Compliance • Risk Management • Automated Monitoring • Industrial Safety • Accident Prevention  
**Industry:** Construction • Manufacturing • Industrial Safety • Mining • Oil & Gas • Workplace Safety  
**Project Type:** Computer Vision & Deep Learning | Industry: Industrial Safety | Focus: Automated Safety Compliance & Risk Reduction

---

## Overview  
This project focuses on building a deep learning–based computer vision system to automatically detect whether workers are wearing safety helmets in industrial or construction environments. The solution improves workplace safety monitoring by automating compliance checks and reducing reliance on manual supervision.

## Objective  
The primary goal was to develop an image classification model capable of distinguishing between workers with and without helmets. Such a system enhances safety enforcement, reduces accident risks, and supports real-time safety monitoring at scale.

---

## Dataset  
- **Source:** Provided as part of the project coursework  
- **Size:** 631 labeled images  
- **Categories:**  
  - `With Helmet` – Workers wearing helmets  
  - `Without Helmet` – Workers without helmets

---

## Workflow  
1. **Data Preprocessing** – Converted images to grayscale, normalized pixel values, and split data into training, validation, and test sets.  
2. **Model Development** – Built and trained CNN-based classifiers, including a baseline CNN and transfer learning models (VGG-16).  
3. **Model Enhancement** – Applied data augmentation, fine-tuned architectures, and compared model performances.  
4. **Evaluation & Insights** – Selected the best-performing model for deployment in real-world safety applications.

---

## Results & Key Insights  
- Delivered a highly accurate model for helmet detection across diverse real-world conditions.  
- Enabled scalable, automated safety compliance monitoring.  
- Demonstrated the potential of computer vision in workplace safety and industrial automation.

---

## Tech Stack  
- **Language:** Python  
- **Libraries:** TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Seaborn  
- **Tools:** Jupyter Notebook / Google Colab  

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Jupyter Notebook or Google Colab
- Required libraries (see requirements below)

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/sandesha21/helmnet-helmet-detection.git
cd helmnet-helmet-detection

# Install required packages
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn

# Launch Jupyter Notebook
jupyter notebook HelmNet_Full_Code_sbadwaik_Final.ipynb
```

### Usage
1. Open the main notebook: `HelmNet_Full_Code_sbadwaik_v2.ipynb` (recommended - enhanced version)
   - Or use `HelmNet_Full_Code_sbadwaik_v1.ipynb` for baseline implementation
2. Run all cells to reproduce the complete analysis
3. The notebook includes data preprocessing, model training, and evaluation
4. Pre-processed data (`images_proj.npy`) and labels (`Labels_proj.csv`) are ready to use

---

## 📊 Model Performance

The trained CNN model achieves:
- **High accuracy** in helmet detection across diverse conditions
- **Robust performance** with data augmentation techniques
- **Transfer learning optimization** using VGG-16 architecture
- **Real-world applicability** for industrial safety monitoring

*Detailed performance metrics and evaluation results are available in the notebook.*

---

## 🔧 Troubleshooting

### Common Issues & Solutions

**Issue: ModuleNotFoundError for TensorFlow/OpenCV**
- Solution: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue: Memory error when loading images_proj.npy**
- Solution: The dataset is large (~631 images). Ensure you have at least 4GB RAM available or use Google Colab for cloud processing

**Issue: Notebook kernel crashes during model training**
- Solution: Reduce batch size in the notebook or use GPU acceleration (Google Colab with GPU runtime)

**Issue: CUDA/GPU not detected**
- Solution: Install GPU-enabled TensorFlow: `pip install tensorflow[and-cuda]` or use CPU-only version

### Getting Help

- Check the [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md) for detailed technical documentation
- Review the Jupyter notebook for inline comments and explanations
- Open an issue on GitHub for bugs or feature requests

---

## 📁 File Structure

```
├── HelmNet_Full_Code_sbadwaik_v1.ipynb                 # Initial CNN model implementation (baseline)
├── HelmNet_Full_Code_sbadwaik_v2.ipynb                 # Enhanced version with VGG-16, data augmentation & advanced evaluation
├── images_proj.npy                                     # Preprocessed image dataset (631 helmet/no-helmet images)
├── Labels_proj.csv                                     # Image classification labels (helmet detection ground truth)
├── PROJECT_DESCRIPTION.md                              # Detailed technical documentation and business context
├── README.md                                           # Project overview and setup guide
└── LICENSE                                             # Project license information
```

---

## 📈 Version Comparison

### **v1 - Baseline Implementation**
- Basic CNN architecture from scratch
- Simple data preprocessing (grayscale, normalization)
- Train/validation/test split
- Standard evaluation metrics (accuracy, confusion matrix)
- Suitable for understanding fundamentals

### **v2 - Enhanced & Production-Ready** ⭐ (Recommended)
- **Multiple model architectures:**
  - Simple CNN (baseline)
  - VGG-16 transfer learning (base model)
  - VGG-16 + custom FFNN layers
  - VGG-16 + FFNN + data augmentation
- **Advanced data augmentation** (rotation, scaling, flipping)
- **Comprehensive evaluation:**
  - Confusion matrices for all models
  - Classification reports (precision, recall, F1-score)
  - Model performance comparisons
  - Detailed observations and insights
- **Better documentation** with business context and recommendations
- **Production-ready** with optimized hyperparameters

**Recommendation:** Use **v2** for deployment and analysis. It provides superior performance through transfer learning and data augmentation techniques.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author  
**Sandesh S. Badwaik**  
*Applied Data Scientist & Machine Learning Engineer*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sbadwaik/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sandesha21)


🌟 **If you found this project helpful, please give it a ⭐!**

---

## 📚 Related Resources & References

- [TensorFlow Documentation](https://www.tensorflow.org/docs)
- [Keras API Reference](https://keras.io/api/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [Computer Vision for Safety Applications](https://arxiv.org/list/cs.CV/recent)
- [Deep Learning Best Practices](https://cs231n.github.io/)


