# CNN_CPAP_CLUSTERS


This repository contains two Jupyter notebooks focused on analyzing CPAP images using deep learning and clustering techniques.


### Files:
1. **CNN_CPAP_CLUSTERS.ipynb**: 
   - This notebook extracts embeddings from CPAP images using a pre-trained CNN model (ResNet50).
   - It then applies K-means clustering to group the images and visualizes the resulting clusters.
   - The goal of this notebook is to perform clustering analysis and explore the grouping of the images based on their features.


2. **CNN_CPAP_Embeddings_&_Cluster_Optimization.ipynb**:
   - This notebook extends the first by not only performing clustering but also using t-SNE to visualize the extracted embeddings in 2D.
   - It further optimizes the number of clusters (k) for K-means by evaluating various metrics such as inertia, silhouette score, and average pairwise distance within clusters.
   - The optimal k is identified, balancing these factors to improve clustering results.


### Overview:
- **CNN-Based Embeddings**: Both notebooks leverage ResNet50 to extract embeddings from CPAP images, transforming them into high-dimensional feature vectors.
- **Clustering**: K-means clustering is used to segment the images into groups based on their embeddings.
- **Visualization**: The second notebook uses t-SNE to reduce dimensionality and visualize the embeddings in a 2D space, while the first focuses on clustering the images directly.
- **Optimization**: The second notebook introduces an optimization process to find the best cluster size using multiple evaluation metrics.


### Requirements:
- Python 3.x
- TensorFlow
- scikit-learn
- matplotlib
- numpy
- PIL


To run the notebooks, simply execute them in a Jupyter environment, ensuring that the necessary libraries are installed.


### License:
This project is licensed under the MIT License - see the LICENSE file for details.