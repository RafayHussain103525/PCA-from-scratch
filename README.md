PCA from Scratch on the Iris Dataset

This project provides a comprehensive, from-scratch implementation of Principal Component Analysis (PCA) using Python. The goal is to demonstrate the inner workings of PCA for dimensionality reduction and data visualization without relying on high-level libraries like Scikit-learn. The famous Iris dataset is used as a case study.

📜 Project Overview

Principal Component Analysis is a fundamental technique in data science and machine learning for reducing the dimensionality of a dataset while preserving as much of its variance as possible. This notebook walks through every step of the PCA algorithm, including:

Data loading and preprocessing.

Calculation of the covariance matrix.

Eigen-decomposition using the Power Iteration method.

Projection of data onto the principal components.

Visualization and interpretation of the results.

✨ Key Features

Scratch Implementation of PCA: The core pca function is built from the ground up using only NumPy for numerical operations.

Step-by-Step Data Preprocessing: Demonstrates how to load, clean, and numerically encode the Iris dataset.

Detailed Statistical Analysis: Includes calculation and discussion of the mean vector, covariance matrix, and explained variance.

Eigen-decomposition via Power Iteration: Implements the Power Iteration algorithm with deflation to find the top k eigenvectors and eigenvalues, providing a look into numerical methods for eigen-decomposition.

Rich Visualizations: Generates several plots to analyze the data and interpret the PCA results:

Feature Distribution Histograms

Feature Relationship Pair Plots

Explained Variance Scree Plot

A comprehensive Biplot showing both the projected data (scores) and original feature influence (loadings).

A Heatmap of feature loadings on the principal components.

🛠️ Methodology: The PCA Algorithm

The PCA process implemented in this notebook follows these key mathematical steps:

Data Loading and Preprocessing: The Iris dataset is loaded from a CSV file. The categorical species labels ('setosa', 'versicolor', 'virginica') are numerically encoded into integers (0, 1, 2) for computational analysis.

Data Centering: The mean of each feature column is calculated to find the dataset's centroid. This mean is then subtracted from every data point. This step is crucial because PCA is designed to find the directions of maximum variance, and variance is measured around the mean.

Covariance Matrix Calculation: The covariance matrix is computed from the centered data. This matrix is a d x d symmetric matrix (where d is the number of features) that describes the variance of each feature and the covariance between each pair of features. It represents the structure and relationships within the data.

Eigen-decomposition: The principal components (the new axes of the data) are the eigenvectors of the covariance matrix. The amount of variance captured by each principal component is its corresponding eigenvalue. This project uses an iterative numerical method to find them:

Power Iteration: An algorithm to find the dominant eigenvector (the one with the largest eigenvalue).

Deflation: A technique to remove the influence of the found eigenvector from the matrix, allowing the Power Iteration to find the next largest eigenvector.

Rayleigh Quotient: Used to calculate the eigenvalue associated with each found eigenvector.

Projection: The original, centered data is projected onto the new axes (the top eigenvectors) to obtain the lower-dimensional representation. This is achieved by taking the dot product of the centered data and the matrix of eigenvectors.

📊 Visualizations & Results
Feature Distributions and Relationships

Initial exploratory analysis showed that "petal length" and "petal width" have a clear bimodal distribution, suggesting the presence of distinct clusters in the data, which PCA is well-suited to capture.

Explained Variance (Scree Plot)

The scree plot shows the proportion of the total variance explained by each principal component.
![image](https://github.com/user-attachments/assets/1044bcf3-0c17-43ba-a906-66fdcd1ebb9a)

PC1 explains 94.6% of the variance.

PC2 explains 5.4% of the variance.

Together, the top two components capture over 97% of the total variance in the original 4-dimensional dataset, confirming that a 2D representation is highly effective.

PCA Biplot

The biplot is the final visualization, showing both the projected data points (scores) and the original feature directions (loadings).
![download](https://github.com/user-attachments/assets/e962e47d-8926-4cfb-bde3-4bc6f38b7f10)


Scores (Dots): The plot shows excellent separation of the three species. The setosa class is completely isolated, while versicolor and virginica are also largely distinct.


Loadings (Arrows): The arrows show how the original features map to the new PC axes.

Petal Length, Petal Width, and Sepal Length contribute most significantly to PC1, which can be interpreted as a general "size" component.

Sepal Width is the primary driver of PC2 and has an inverse relationship with the other features along PC1.

Feature Loadings Heatmap

This heatmap provides a clear, quantitative view of the feature loadings.
![download](https://github.com/user-attachments/assets/5292a7e6-24ed-4c33-9a47-a130469cd59a)

🚀 How to Run

Clone this repository:

Generated bash
git clone <repository-url>


Ensure you have the required dependencies installed (see below).

Place the Iris.csv file in the correct path as specified in the notebook.

Open and run the PCA_scratch.ipynb notebook in a Jupyter environment.

📦 Dependencies

Python 3.x

numpy

matplotlib

seaborn

Discussion: Why Power Iteration instead of the Characteristic Polynomial?

A common question when implementing PCA from scratch is how to find the eigenvalues and eigenvectors of the covariance matrix. The standard "textbook" method involves solving the characteristic polynomial, but this project uses an iterative numerical method called Power Iteration with Deflation. Here’s a detailed explanation of why this choice was made.

1. The "Textbook" Method: The Characteristic Polynomial

The mathematical definition of an eigenvalue (λ) and its corresponding eigenvector (v) for a matrix A is:

Av = λv

To solve this, we can rearrange it to:

(A - λI)v = 0

For this equation to have a non-zero solution for v, the matrix (A - λI) must be singular, which means its determinant must be zero:

det(A - λI) = 0

This equation is called the characteristic polynomial. For a d x d matrix, this determinant expands into a degree-d polynomial in the variable λ. The roots of this polynomial are the eigenvalues of the matrix A.

Why This Method Is Not Ideal for Computation:

While elegant in theory, solving the characteristic polynomial is rarely used in practical, computational applications for two main reasons:

Numerical Instability: Finding the roots of a polynomial is a notoriously ill-conditioned problem. This means that very small errors in the polynomial's coefficients (which are inevitable due to floating-point arithmetic) can lead to very large errors in the calculated roots (the eigenvalues). For matrices larger than 4x4, this instability becomes a significant issue. In fact, the Abel-Ruffini theorem proves that there is no general algebraic formula for the roots of polynomials of degree five or higher, meaning their roots must be found numerically anyway.

Computational Inefficiency for PCA: For PCA, we are almost always interested in only the top k principal components—those corresponding to the largest eigenvalues. The characteristic polynomial method requires you to find all d eigenvalues first, and then sort them to find the largest ones. This is highly inefficient if d is large and you only need a few components.

2. The Practical Method: Power Iteration with Deflation and the Rayleigh Quotient

This project uses an iterative numerical method, which is far more common in real-world software.

How It Works:

Power Iteration (Finding the Dominant Eigenvector): This algorithm finds the eigenvector corresponding to the largest eigenvalue.

It starts with a random unit vector v.

It repeatedly multiplies this vector by the covariance matrix A: v_next = A @ v.

After each multiplication, it normalizes the vector back to unit length: v_next = v_next / ||v_next||.

With each iteration, the vector v aligns more and more with the direction of the dominant eigenvector. This is because the component of the random vector in this direction is amplified more than any other component in each multiplication. The process stops when the direction of v no longer changes significantly.

Rayleigh Quotient (Finding the Eigenvalue): Once the eigenvector v is found, its corresponding eigenvalue λ is efficiently calculated using the Rayleigh Quotient:

λ = v.T @ A @ v
(This works because v is a unit vector).

Deflation (Finding Subsequent Eigenvectors): Power Iteration only finds the eigenvector with the largest eigenvalue. To find the second principal component, we must "deflate" the covariance matrix by removing the influence of the first component we just found. This is done with the following update:

A_next = A - λ * np.outer(v, v)

This new matrix A_next has the same eigenvectors and eigenvalues as the original A, except that the eigenvalue corresponding to v is now zero. When we run Power Iteration again on A_next, it will converge to the next largest eigenvector. This process is repeated k times to find the top k components.

Why This Method Is a Better Choice:

Numerical Stability: Iterative methods like Power Iteration are generally much more stable than root-finding for polynomials. They work directly with the matrix and are less susceptible to the cascading errors of the polynomial approach.

Efficiency for PCA: This method is perfectly suited for PCA. It naturally finds the eigenvectors in descending order of their eigenvalues' magnitude. If you only need the top 2 components, you only need to run the loop twice, making it far more efficient than finding all four eigenvalues.

Conceptual Insight: Implementing Power Iteration provides a deeper understanding of what eigenvectors represent: they are the directions that remain invariant (up to scaling) under the linear transformation defined by the matrix.
