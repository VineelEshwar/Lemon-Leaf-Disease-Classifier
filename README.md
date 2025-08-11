


# 🍋 Lemon Leaf Disease Classifier

A **Streamlit-based web application** for detecting and classifying diseases in lemon leaves using deep learning.  
Simply upload an image of a lemon leaf, and the model will predict the disease type along with treatment and fertilizer recommendations.

---

## 🌐 Live Demo
🔗 **Deployed App:** [Lemon Leaf Disease Classifier](https://lemon-leaf-disease-classifiers.streamlit.app/)  

📄 **Reference Paper:** [Lemon Leaf Disease Recognition Using Vision Transformer Networks(IEEE)](https://ieeexplore.ieee.org/document/11064055)

---

## 📌 Features
- Upload **JPG, JPEG, or PNG** images of lemon leaves.
- Get **instant predictions** about the health of the leaf.
- Provides **treatment suggestions** and **fertilizer advice**.
- Simple, fast, and accessible from any device.
- Deployed entirely on **Streamlit Cloud**.

---



---

## 🧠 Model Information
- The model is based on **EfficientNet** architecture.
- Trained on a lemon leaf disease dataset for classification.
- Preprocessing includes resizing to 224×224 pixels, normalization, and tensor conversion.

---

## 📂 Project Structure
```


├── app/
│   ├── app.py      # Main Streamlit application
│   ├── model.py                   # Model definition
│   ├── utils.py              # Image preprocessing and helper functions
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── saved\_model.h5            # Pre-trained model weights

````

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/lemon-leaf-disease-classifier.git
cd lemon-leaf-disease-classifier
````

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

```bash
streamlit run app/app.py
```

The app will run at: `http://localhost:8501`

---

## ⚙️ Requirements

* Python 3.10+
* Streamlit
* PyTorch & Torchvision
* Pillow
* NumPy (<2.0 for compatibility)

---

## 📚 Reference

If you want to understand the underlying research, read the paper:
**[	Lemon Leaf Disease Recognition Using Vision Transformer Networks (IEEE)](https://ieeexplore.ieee.org/document/11064055)**

---

## 📜 License

This project is licensed under the **MIT License** - feel free to use and modify it.

---

## 💡 Acknowledgments

* [Streamlit](https://streamlit.io/) for making deployment easy.
* [IEEE Xplore](https://ieeexplore.ieee.org/) for research access.
* The open-source ML community for tools and datasets.

---


