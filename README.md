# Deep-Learning-Based-Brain-Tumor-Analysis-Using-MRIImages
Deep learning-based brain tumor Analysis using MRI images, where tumor regions are segmented using 3D U-Net and classified using a ResNet model. The pipeline improves accuracy by focusing classification on segmented tumor regions, enabling effective localization and reliable prediction of tumor types.

⚙️ Methodology
🔹 1. Preprocessing
MRI normalization
Resizing and noise removal
Preparation of 3D volumetric data
🔹 2. Tumor Segmentation
Model: 3D U-Net
Input: 3D MRI volumes
Output: Tumor mask (voxel-wise segmentation)
Captures spatial and structural tumor information
🔹 3. Tumor Region Extraction
Segmentation mask applied to MRI
Tumor region is isolated
Reduces background noise
🔹 4. Tumor Classification
Model: ResNet (CNN)
Input: Extracted tumor region / slices
Output: Tumor type:
Glioma
Meningioma
Pituitary
Learns deep features for accurate classification

🧠 Model Architecture
3D U-Net → segmentation (encoder-decoder with skip connections)
ResNet → classification (residual learning for deep feature extraction)

📊 Dataset
🔹 Segmentation Dataset
BraTS Dataset
Multi-modal MRI scans:
T1
T2
FLAIR
Used for training the 3D U-Net model
🔹 Classification Dataset
Brain tumor MRI dataset (Kaggle or similar)
Contains labeled tumor types:
Glioma
Meningioma
Pituitary
Used for training the ResNet model

🚀 Features
Accurate tumor segmentation using 3D U-Net
Improved classification using extracted tumor regions
Handles 3D MRI data effectively
End-to-end deep learning pipeline

🛠️ Technologies Used
Python
PyTorch / TensorFlow
NumPy, OpenCV, Matplotlib

📈 Results
Precise tumor localization
Improved classification accuracy
Better performance compared to direct classification

📌 Future Work
Model optimization and tuning
Integration with clinical systems

📖 References
BraTS Dataset
Deep Learning for Medical Imaging
U-Net and ResNet architectures

👩‍💻 Author
Shreeja Raju
