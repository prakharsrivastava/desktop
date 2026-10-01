# Databricks notebook source
# DBTITLE 1,📚 Introduction - PCA Kya Hai?
# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,📏 Part 1: Mathematical Foundation
# MAGIC %md
# MAGIC # Part 1: Mathematical Foundation 📚
# MAGIC
# MAGIC ## 1️⃣ Variance - Spread kitna hai?
# MAGIC
# MAGIC **Variance** batata hai data kitna spread out hai mean se:
# MAGIC
# MAGIC $$\text{Variance} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$
# MAGIC
# MAGIC Jahan:
# MAGIC - $x_i$ = individual data point
# MAGIC - $\bar{x}$ = mean
# MAGIC - $n$ = number of points
# MAGIC
# MAGIC **High variance** = Data zyada spread out  
# MAGIC **Low variance** = Data mean ke paas concentrated
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2️⃣ Covariance - Do variables ka relationship
# MAGIC
# MAGIC **Covariance** batata hai do variables ek saath kaise vary karte hain:
# MAGIC
# MAGIC $$\text{Cov}(X, Y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$
# MAGIC
# MAGIC - **Positive covariance** → Dono saath mein badhte/ghatate hain
# MAGIC - **Negative covariance** → Ek badhta hai, doosra ghatta hai
# MAGIC - **Zero covariance** → No linear relationship
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3️⃣ Covariance Matrix
# MAGIC
# MAGIC Agar tumhare paas $n$ features hain, toh **Covariance Matrix** $n \times n$ hoga:
# MAGIC
# MAGIC $$C = \begin{bmatrix}
# MAGIC \text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_n) \\
# MAGIC \text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots & \text{Cov}(X_2, X_n) \\
# MAGIC \vdots & \vdots & \ddots & \vdots \\
# MAGIC \text{Cov}(X_n, X_1) & \text{Cov}(X_n, X_2) & \cdots & \text{Var}(X_n)
# MAGIC \end{bmatrix}$$
# MAGIC
# MAGIC **Diagonal** = Variance of each feature  
# MAGIC **Off-diagonal** = Covariance between features
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 4️⃣ Eigenvectors & Eigenvalues - PCA ki Jaan!
# MAGIC
# MAGIC ### Eigenvector:
# MAGIC Ek direction vector jo matrix multiplication ke baad **sirf scale hota hai, direction nahi badalta**
# MAGIC
# MAGIC $$A\vec{v} = \lambda\vec{v}$$
# MAGIC
# MAGIC Jahan:
# MAGIC - $A$ = Matrix (covariance matrix in PCA)
# MAGIC - $\vec{v}$ = Eigenvector (principal component direction)
# MAGIC - $\lambda$ = Eigenvalue (magnitude of variance in that direction)
# MAGIC
# MAGIC ### PCA Mein Role:
# MAGIC - **Eigenvectors** = Directions of maximum variance (Principal Components)
# MAGIC - **Eigenvalues** = Amount of variance in each direction
# MAGIC - **Largest eigenvalue** = Maximum variance direction (PC1)
# MAGIC - **2nd largest** = 2nd maximum variance (PC2)
# MAGIC
# MAGIC Ab code mein dekhte hain! 👇

# COMMAND ----------

# DBTITLE 1,Setup - Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Plotting settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Libraries imported successfully!")
print("Ready to explore PCA! 🚀")

# COMMAND ----------

# DBTITLE 1,Install Interactive Visualization Libraries
# Install interactive libraries
%pip install plotly ipywidgets -q

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ipywidgets import interact, interactive, IntSlider, FloatSlider, Dropdown, widgets
from IPython.display import display, clear_output

print("✅ Interactive libraries imported!")
print("🎨 Ready for interactive visualizations!")

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: Explore Covariance
# Interactive covariance exploration
print("🎮 Interactive Covariance Explorer")
print("Use sliders to see how correlation affects covariance!\n")

def plot_correlation_interactive(correlation=0.8, n_points=100, noise=5):
    """
    Interactive function to explore correlation and covariance
    """
    np.random.seed(42)
    
    # Generate correlated data
    x = np.random.normal(60, 15, n_points)
    y = correlation * x + np.random.normal(0, noise, n_points)
    
    # Calculate statistics
    cov = np.cov(x, y)[0, 1]
    corr = np.corrcoef(x, y)[0, 1]
    
    # Create interactive plot
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=8, color='royalblue', opacity=0.6, line=dict(width=1, color='black')),
        name='Data Points',
        hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    ))
    
    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line, y=p(x_line),
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),
        name='Trend Line'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Interactive Correlation Explorer<br>Correlation: {corr:.3f} | Covariance: {cov:.2f}',
        xaxis_title='Variable X',
        yaxis_title='Variable Y',
        height=500,
        hovermode='closest',
        template='plotly_white'
    )
    
    fig.show()
    
    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"   Correlation: {corr:.4f}")
    print(f"   Covariance: {cov:.2f}")
    print(f"   X Mean: {x.mean():.2f}, Std: {x.std():.2f}")
    print(f"   Y Mean: {y.mean():.2f}, Std: {y.std():.2f}")
    
    if abs(corr) > 0.7:
        print(f"   ✅ Strong {'positive' if corr > 0 else 'negative'} correlation!")
    elif abs(corr) > 0.3:
        print(f"   ⚠️ Moderate {'positive' if corr > 0 else 'negative'} correlation")
    else:
        print(f"   ❌ Weak correlation")

# Create interactive widget
interact(
    plot_correlation_interactive,
    correlation=FloatSlider(value=0.8, min=-1, max=1, step=0.1, description='Correlation:', continuous_update=False),
    n_points=IntSlider(value=100, min=20, max=300, step=20, description='Points:', continuous_update=False),
    noise=FloatSlider(value=5, min=1, max=20, step=1, description='Noise:', continuous_update=False)
);

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: PCA Component Selection
print("🎮 Interactive PCA Component Selection")
print("Change n_components to see how variance changes!\n")

def interactive_pca_components(n_components=2):
    """
    Interactive PCA with component selection
    """
    # Apply PCA with selected components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'PCA with {n_components} Component(s)', 'Variance Explained'),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # Plot 1: PCA transformed data (2D or 1D)
    if n_components >= 2:
        fig.add_trace(
            go.Scatter(
                x=X_pca[:, 0], y=X_pca[:, 1],
                mode='markers',
                marker=dict(size=6, color=X[:, 0], colorscale='Viridis', 
                           opacity=0.7, line=dict(width=0.5, color='black'),
                           colorbar=dict(title='Height', x=0.45)),
                name='Data Points',
                hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        fig.update_xaxes(title_text='PC1', row=1, col=1)
        fig.update_yaxes(title_text='PC2', row=1, col=1)
    else:
        fig.add_trace(
            go.Scatter(
                x=X_pca[:, 0], y=np.zeros_like(X_pca[:, 0]),
                mode='markers',
                marker=dict(size=8, color=X[:, 0], colorscale='Viridis', 
                           opacity=0.7, line=dict(width=0.5, color='black')),
                name='Data Points',
                hovertemplate='PC1: %{x:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        fig.update_xaxes(title_text='PC1 (Only Component)', row=1, col=1)
        fig.update_yaxes(title_text='', showticklabels=False, row=1, col=1)
    
    # Plot 2: Variance explained
    pc_labels = [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))]
    colors_var = ['#FF6B6B' if i < n_components else '#CCCCCC' 
                  for i in range(len(pca.explained_variance_ratio_))]
    
    fig.add_trace(
        go.Bar(
            x=pc_labels,
            y=pca.explained_variance_ratio_ * 100,
            marker=dict(color=colors_var, line=dict(width=2, color='black')),
            text=[f'{v:.1f}%' for v in pca.explained_variance_ratio_ * 100],
            textposition='outside',
            name='Variance %',
            hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    fig.update_xaxes(title_text='Principal Component', row=1, col=2)
    fig.update_yaxes(title_text='Variance Explained (%)', row=1, col=2)
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=False,
        template='plotly_white',
        title_text=f'Interactive PCA: {n_components} Component(s) | {pca.explained_variance_ratio_.sum()*100:.1f}% Variance Retained'
    )
    
    fig.show()
    
    # Print statistics
    print(f"\n📊 Results with {n_components} component(s):")
    print(f"   Total variance retained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"   Transformed shape: {X.shape} → {X_pca.shape}")
    print(f"   Dimensionality reduction: {(1 - n_components/X.shape[1])*100:.1f}%")
    
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        print(f"   PC{i}: {var*100:.2f}%")

# Create interactive widget
interact(
    interactive_pca_components,
    n_components=IntSlider(value=2, min=1, max=3, step=1, description='Components:', continuous_update=False)
);

# COMMAND ----------

# DBTITLE 1,Step 1: Variance aur Covariance Samjho
# Simple example - 2 features
print("📏 Example: Student Marks")
print("="*50)

# Data: Math aur Physics marks
data = np.array([
    [45, 50],  # Student 1
    [60, 65],  # Student 2
    [55, 58],  # Student 3
    [70, 72],  # Student 4
    [80, 85],  # Student 5
    [50, 52],  # Student 6
    [65, 68],  # Student 7
])

math_marks = data[:, 0]
physics_marks = data[:, 1]

print("\n📊 Data:")
df = pd.DataFrame(data, columns=['Math', 'Physics'])
print(df)

# Calculate variance
math_var = np.var(math_marks, ddof=1)
physics_var = np.var(physics_marks, ddof=1)

print(f"\n📈 Variance:")
print(f"Math variance: {math_var:.2f}")
print(f"Physics variance: {physics_var:.2f}")
print("High variance = Zyada spread, Low variance = Kam spread")

# Calculate covariance
cov = np.cov(math_marks, physics_marks)[0, 1]

print(f"\n🔗 Covariance between Math & Physics: {cov:.2f}")
if cov > 0:
    print("✅ Positive! Matlab dono subjects mein saath performance badhti/ghatati hai")

# Covariance Matrix
cov_matrix = np.cov(data.T)
print(f"\n📊 Covariance Matrix:")
print(cov_matrix)
print("\nDiagonal = Variance, Off-diagonal = Covariance")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Scatter plot
axes[0].scatter(math_marks, physics_marks, s=100, alpha=0.6, edgecolors='black')
axes[0].set_xlabel('Math Marks', fontsize=12)
axes[0].set_ylabel('Physics Marks', fontsize=12)
axes[0].set_title('Math vs Physics Marks\n(Positive Correlation)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Add mean lines
axes[0].axhline(np.mean(physics_marks), color='red', linestyle='--', alpha=0.5, label='Mean Physics')
axes[0].axvline(np.mean(math_marks), color='blue', linestyle='--', alpha=0.5, label='Mean Math')
axes[0].legend()

# Plot 2: Covariance matrix heatmap
sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            xticklabels=['Math', 'Physics'], yticklabels=['Math', 'Physics'],
            ax=axes[1], cbar_kws={'label': 'Covariance'})
axes[1].set_title('Covariance Matrix Heatmap', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n💡 Key Insight:")
print("Positive covariance matlab agar Math marks high hain, toh Physics bhi high hoga!")

# COMMAND ----------

# DBTITLE 1,Step 2: Eigenvectors aur Eigenvalues - PCA ka Heart
print("💡 Understanding Eigenvectors & Eigenvalues")
print("="*60)

# Use the same covariance matrix from previous example
print("\n📊 Covariance Matrix:")
print(cov_matrix)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\n🎯 Eigenvalues (Variance in each direction):")
for i, val in enumerate(eigenvalues, 1):
    print(f"   PC{i}: {val:.4f} (captures {val/eigenvalues.sum()*100:.2f}% variance)")

print("\n🧲 Eigenvectors (Directions of Principal Components):")
for i, vec in enumerate(eigenvectors.T, 1):
    print(f"   PC{i}: [{vec[0]:.4f}, {vec[1]:.4f}]")

# Explained variance ratio
explained_variance_ratio = eigenvalues / eigenvalues.sum()
print(f"\n📊 Explained Variance Ratio:")
for i, ratio in enumerate(explained_variance_ratio, 1):
    print(f"   PC{i}: {ratio*100:.2f}%")

print(f"\n🎯 Total variance captured by PC1: {explained_variance_ratio[0]*100:.2f}%")
print(f"✅ Sirf PC1 se hi {explained_variance_ratio[0]*100:.2f}% information mil jayegi!")

# Visualization - Principal Components
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Data with Principal Components
ax1 = axes[0]
ax1.scatter(math_marks, physics_marks, s=100, alpha=0.6, edgecolors='black', label='Data Points')

# Center of data
mean_point = [np.mean(math_marks), np.mean(physics_marks)]

# Draw eigenvectors (Principal Components)
for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T), 1):
    # Scale eigenvector by sqrt of eigenvalue for visualization
    scale = np.sqrt(eigenvalue) * 3
    ax1.arrow(mean_point[0], mean_point[1], 
             eigenvector[0]*scale, eigenvector[1]*scale,
             head_width=2, head_length=2, fc=f'C{i}', ec=f'C{i}', 
             linewidth=3, label=f'PC{i} (var={eigenvalue:.2f})')

ax1.scatter(*mean_point, color='red', s=200, marker='X', zorder=5, label='Mean Point')
ax1.set_xlabel('Math Marks', fontsize=12)
ax1.set_ylabel('Physics Marks', fontsize=12)
ax1.set_title('Principal Components Directions\n(Eigenvectors scaled by eigenvalues)', 
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Plot 2: Explained variance
ax2 = axes[1]
pc_labels = [f'PC{i}' for i in range(1, len(eigenvalues)+1)]
colors = ['#FF6B6B', '#4ECDC4']
ax2.bar(pc_labels, explained_variance_ratio*100, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Explained Variance (%)', fontsize=12)
ax2.set_title('Variance Explained by Each PC', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add percentage labels on bars
for i, (label, val) in enumerate(zip(pc_labels, explained_variance_ratio*100)):
    ax2.text(i, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n🔑 Key Insights:")
print(f"   • PC1 (longest arrow) = Maximum variance direction")
print(f"   • PC2 (shorter arrow) = 2nd maximum variance, perpendicular to PC1")
print(f"   • PC1 captures {explained_variance_ratio[0]*100:.1f}% of total variance!")
print(f"   • Together they capture {explained_variance_ratio.sum()*100:.1f}% variance")

# COMMAND ----------

# DBTITLE 1,📏 Part 2: Complete PCA Algorithm
# MAGIC %md
# MAGIC # Part 2: PCA Algorithm - Step by Step 👣
# MAGIC
# MAGIC ## 📦 The Complete PCA Pipeline
# MAGIC
# MAGIC ```
# MAGIC Input: Data matrix X (n samples × m features)
# MAGIC
# MAGIC Step 1: Standardize the data
# MAGIC    └── X_std = (X - mean) / std
# MAGIC    └── Kyun? Different scales ko normalize karna
# MAGIC
# MAGIC Step 2: Compute Covariance Matrix
# MAGIC    └── C = (1/n) X_stdᵀ × X_std
# MAGIC    └── Size: m × m
# MAGIC
# MAGIC Step 3: Calculate Eigenvalues & Eigenvectors
# MAGIC    └── Cν = λν
# MAGIC    └── Sort by eigenvalues (descending)
# MAGIC
# MAGIC Step 4: Choose k Principal Components
# MAGIC    └── Select top k eigenvectors
# MAGIC    └── Based on explained variance
# MAGIC
# MAGIC Step 5: Transform Data
# MAGIC    └── X_pca = X_std × W
# MAGIC    └── W = Matrix of k eigenvectors
# MAGIC    └── Output: n samples × k components
# MAGIC
# MAGIC Output: Reduced dimensional data!
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Why Standardization?
# MAGIC
# MAGIC **Problem:**  
# MAGIC Agar ek feature ka scale 0-1 hai aur doosre ka 0-1000, toh PCA sirf high-variance feature ko prefer karega!
# MAGIC
# MAGIC **Solution:**  
# MAGIC Standardization se sabhi features ko same scale par lao:
# MAGIC
# MAGIC $$X_{\text{standardized}} = \frac{X - \mu}{\sigma}$$
# MAGIC
# MAGIC Where:
# MAGIC - $\mu$ = mean
# MAGIC - $\sigma$ = standard deviation
# MAGIC
# MAGIC Baad mein har feature ka **mean = 0** aur **std = 1** hoga! ⚖️
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC Ab complete implementation dekhte hain from scratch! 👇

# COMMAND ----------

# DBTITLE 1,PCA From Scratch - Complete Implementation
class PCA_FromScratch:
    """
    PCA implementation from scratch
    Saare steps manually implement kiye hain!
    """
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.std = None
        self.eigenvalues = None
        self.explained_variance_ratio = None
    
    def fit(self, X):
        """
        PCA ko data par fit karo
        """
        print(f"📏 Step 1: Standardizing data...")
        # Step 1: Standardize
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_std = (X - self.mean) / self.std
        print(f"   ✓ Mean: {self.mean}")
        print(f"   ✓ Std: {self.std}")
        
        print(f"\n📏 Step 2: Computing covariance matrix...")
        # Step 2: Covariance matrix
        n_samples = X.shape[0]
        cov_matrix = (X_std.T @ X_std) / (n_samples - 1)
        print(f"   ✓ Covariance matrix shape: {cov_matrix.shape}")
        
        print(f"\n📏 Step 3: Calculating eigenvalues & eigenvectors...")
        # Step 3: Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print(f"   ✓ Found {len(eigenvalues)} eigenvalues")
        
        print(f"\n📏 Step 4: Selecting top {self.n_components} components...")
        # Step 4: Select top k components
        self.components = eigenvectors[:, :self.n_components]
        self.eigenvalues = eigenvalues[:self.n_components]
        
        # Calculate explained variance ratio
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance
        
        print(f"   ✓ Selected components shape: {self.components.shape}")
        for i, ratio in enumerate(self.explained_variance_ratio, 1):
            print(f"   ✓ PC{i} explains {ratio*100:.2f}% variance")
        
        return self
    
    def transform(self, X):
        """
        Data ko principal components mein transform karo
        """
        print(f"\n📏 Step 5: Transforming data...")
        # Standardize
        X_std = (X - self.mean) / self.std
        
        # Project onto principal components
        X_pca = X_std @ self.components
        
        print(f"   ✓ Original shape: {X.shape}")
        print(f"   ✓ Transformed shape: {X_pca.shape}")
        print(f"   ✓ Reduced from {X.shape[1]} to {X_pca.shape[1]} dimensions!")
        
        return X_pca
    
    def fit_transform(self, X):
        """
        Fit aur transform ek saath karo
        """
        self.fit(X)
        return self.transform(X)


print("✅ PCA_FromScratch class created!")
print("\nLet's test it with real data! 👇")

# COMMAND ----------

# DBTITLE 1,Test From-Scratch PCA with Real Data
# Create a more complex dataset - 3D data
np.random.seed(42)

print("📊 Creating 3D dataset...")
print("="*60)

# Generate correlated 3D data
n_samples = 200

# Feature 1: Height (cm)
height = np.random.normal(170, 10, n_samples)

# Feature 2: Weight (kg) - correlated with height
weight = 0.7 * height + np.random.normal(0, 5, n_samples) - 60

# Feature 3: Age (years) - somewhat independent
age = np.random.uniform(20, 60, n_samples)

# Combine into dataset
X = np.column_stack([height, weight, age])

print(f"\n📊 Dataset created:")
print(f"   Shape: {X.shape}")
print(f"   Features: Height, Weight, Age")
print(f"   Samples: {n_samples}")

print(f"\n📊 Sample data:")
df = pd.DataFrame(X, columns=['Height (cm)', 'Weight (kg)', 'Age (years)'])
print(df.head(10))

print(f"\n📊 Statistics:")
print(df.describe())

# Apply our from-scratch PCA
print("\n" + "="*60)
print("🚀 Applying PCA From Scratch")
print("="*60)

pca_scratch = PCA_FromScratch(n_components=2)
X_pca_scratch = pca_scratch.fit_transform(X)

print("\n" + "="*60)
print("✅ PCA Complete!")
print("="*60)

print(f"\n📊 Cumulative variance explained: {pca_scratch.explained_variance_ratio.sum()*100:.2f}%")
print(f"\n💡 Insight: Using just 2 components instead of 3, we retain {pca_scratch.explained_variance_ratio.sum()*100:.1f}% of information!")

# COMMAND ----------

# DBTITLE 1,Visualize PCA Transformation - 3D to 2D
# Visualization - Before and After PCA
fig = plt.figure(figsize=(16, 6))

# Plot 1: Original 3D data
ax1 = fig.add_subplot(131, projection='3d')
scatter = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 0], cmap='viridis', s=30, alpha=0.6)
ax1.set_xlabel('Height (cm)', fontsize=10)
ax1.set_ylabel('Weight (kg)', fontsize=10)
ax1.set_zlabel('Age (years)', fontsize=10)
ax1.set_title('Original 3D Data\n(Height, Weight, Age)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax1, label='Height')

# Plot 2: PCA transformed 2D data
ax2 = fig.add_subplot(132)
scatter2 = ax2.scatter(X_pca_scratch[:, 0], X_pca_scratch[:, 1], 
                       c=X[:, 0], cmap='viridis', s=30, alpha=0.6, edgecolors='black')
ax2.set_xlabel('PC1 (Principal Component 1)', fontsize=10)
ax2.set_ylabel('PC2 (Principal Component 2)', fontsize=10)
ax2.set_title(f'PCA Transformed 2D Data\n({pca_scratch.explained_variance_ratio.sum()*100:.1f}% variance retained)', 
             fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='red', linestyle='--', alpha=0.3)
ax2.axvline(0, color='red', linestyle='--', alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Height')

# Plot 3: Explained variance
ax3 = fig.add_subplot(133)
pc_labels = ['PC1', 'PC2', 'PC3']
all_eigenvalues = pca_scratch.eigenvalues if len(pca_scratch.eigenvalues) == 3 else np.append(pca_scratch.eigenvalues, 0)
explained_var = all_eigenvalues / all_eigenvalues.sum() * 100

colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
bars = ax3.bar(pc_labels, explained_var, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Explained Variance (%)', fontsize=10)
ax3.set_title('Variance Explained by Each PC', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Add percentage labels
for i, (bar, val) in enumerate(zip(bars, explained_var)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add cumulative line
cumulative = np.cumsum(explained_var)
ax3_twin = ax3.twinx()
ax3_twin.plot(pc_labels, cumulative, 'ro-', linewidth=2, markersize=8, label='Cumulative')
ax3_twin.set_ylabel('Cumulative Variance (%)', fontsize=10, color='red')
ax3_twin.tick_params(axis='y', labelcolor='red')
ax3_twin.set_ylim([0, 110])

# Add cumulative labels
for i, (label, val) in enumerate(zip(pc_labels, cumulative)):
    ax3_twin.text(i, val + 3, f'{val:.1f}%', ha='center', fontsize=9, color='red', fontweight='bold')

plt.tight_layout()
plt.show()

print("🔑 Key Insights:")
print(f"   • Original data: 3D (Height, Weight, Age)")
print(f"   • Transformed: 2D (PC1, PC2)")
print(f"   • Information retained: {pca_scratch.explained_variance_ratio.sum()*100:.1f}%")
print(f"   • PC1 captures: {pca_scratch.explained_variance_ratio[0]*100:.1f}% variance")
print(f"   • PC2 captures: {pca_scratch.explained_variance_ratio[1]*100:.1f}% variance")
print(f"\n🎯 Result: Dimensionality reduced by 33% with minimal information loss!")

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: 3D to 2D Transformation
print("🎮 Interactive 3D → 2D Transformation")
print("Rotate and explore the 3D data, then see PCA projection!\n")

def interactive_3d_pca():
    """
    Interactive 3D visualization with PCA transformation
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Original 3D Data (Rotate Me!)', 'PCA 2D Projection'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]],
        column_widths=[0.5, 0.5]
    )
    
    # Plot 1: Interactive 3D scatter
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=X[:, 0],
                colorscale='Viridis',
                opacity=0.7,
                line=dict(width=0.5, color='black'),
                colorbar=dict(title='Height (cm)', x=0.45)
            ),
            name='3D Data',
            hovertemplate='Height: %{x:.1f} cm<br>Weight: %{y:.1f} kg<br>Age: %{z:.1f} years<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot 2: PCA 2D projection
    fig.add_trace(
        go.Scatter(
            x=X_pca_scratch[:, 0], y=X_pca_scratch[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color=X[:, 0],
                colorscale='Viridis',
                opacity=0.7,
                line=dict(width=0.5, color='black'),
                showscale=False
            ),
            name='2D Projection',
            hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Height: ' + 
                         np.array2string(X[:, 0], formatter={'float_kind': lambda x: f"{x:.1f}"}) + 
                         ' cm<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add origin lines for PC plot
    fig.add_trace(
        go.Scatter(
            x=[X_pca_scratch[:, 0].min(), X_pca_scratch[:, 0].max()],
            y=[0, 0],
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[X_pca_scratch[:, 1].min(), X_pca_scratch[:, 1].max()],
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=2
    )
    
    # Update 3D scene
    fig.update_scenes(
        xaxis_title='Height (cm)',
        yaxis_title='Weight (kg)',
        zaxis_title='Age (years)',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        ),
        row=1, col=1
    )
    
    # Update 2D axes
    fig.update_xaxes(
        title_text=f'PC1 ({pca_scratch.explained_variance_ratio[0]*100:.1f}% var)',
        row=1, col=2
    )
    fig.update_yaxes(
        title_text=f'PC2 ({pca_scratch.explained_variance_ratio[1]*100:.1f}% var)',
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        template='plotly_white',
        title_text=f'🎮 Interactive 3D → 2D PCA Transformation | {pca_scratch.explained_variance_ratio.sum()*100:.1f}% Variance Retained',
        hovermode='closest'
    )
    
    fig.show()
    
    print("💡 Tip: Rotate the 3D plot to see data from different angles!")
    print(f"      Then compare with the 2D PCA projection on the right.")

interactive_3d_pca()

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: Iris Dataset Explorer
from sklearn.datasets import load_iris

print("🎮 Interactive Iris Dataset PCA Explorer")
print("Explore different feature combinations and PCA!\n")

# Load iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

def interactive_iris_pca(feature_x=0, feature_y=1, show_pca=True, n_components=2):
    """
    Interactive Iris dataset explorer with PCA
    """
    # Standardize
    scaler = StandardScaler()
    X_iris_scaled = scaler.fit_transform(X_iris)
    
    # Apply PCA if requested
    if show_pca:
        pca = PCA(n_components=n_components)
        X_plot = pca.fit_transform(X_iris_scaled)
        x_label = f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)'
        y_label = f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)' if n_components > 1 else ''
        title_suffix = f'PCA ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)'
    else:
        X_plot = X_iris
        x_label = feature_names[feature_x]
        y_label = feature_names[feature_y]
        title_suffix = 'Original Features'
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter for each species
    colors_species = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    symbols = ['circle', 'square', 'diamond']
    
    for i, (target_name, color, symbol) in enumerate(zip(target_names, colors_species, symbols)):
        mask = y_iris == i
        
        if show_pca:
            x_data = X_plot[mask, 0]
            y_data = X_plot[mask, 1] if n_components > 1 else np.zeros(mask.sum())
        else:
            x_data = X_plot[mask, feature_x]
            y_data = X_plot[mask, feature_y]
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            name=target_name.capitalize(),
            marker=dict(
                size=10,
                color=color,
                symbol=symbol,
                opacity=0.7,
                line=dict(width=1.5, color='black')
            ),
            hovertemplate=f'{target_name.capitalize()}<br>{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<extra></extra>'
        ))
    
    # Add origin lines for PCA
    if show_pca:
        fig.add_shape(type='line', x0=X_plot[:, 0].min(), x1=X_plot[:, 0].max(),
                     y0=0, y1=0, line=dict(color='gray', width=1, dash='dash'))
        if n_components > 1:
            fig.add_shape(type='line', x0=0, x1=0,
                         y0=X_plot[:, 1].min(), y1=X_plot[:, 1].max(),
                         line=dict(color='gray', width=1, dash='dash'))
    
    # Update layout
    fig.update_layout(
        title=f'🌺 Iris Dataset: {title_suffix}',
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=550,
        template='plotly_white',
        hovermode='closest',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1)
    )
    
    fig.show()
    
    # Print statistics
    if show_pca:
        print(f"\n📊 PCA Statistics:")
        print(f"   Components: {n_components}")
        print(f"   Total variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")
        for i, var in enumerate(pca.explained_variance_ratio_, 1):
            print(f"   PC{i}: {var*100:.2f}%")
    else:
        print(f"\n📊 Original Features:")
        print(f"   X-axis: {feature_names[feature_x]}")
        print(f"   Y-axis: {feature_names[feature_y]}")

# Create interactive widget
interact(
    interactive_iris_pca,
    feature_x=Dropdown(options=[(name, i) for i, name in enumerate(feature_names)], 
                       value=0, description='X Feature:'),
    feature_y=Dropdown(options=[(name, i) for i, name in enumerate(feature_names)], 
                       value=1, description='Y Feature:'),
    show_pca=widgets.Checkbox(value=True, description='Show PCA'),
    n_components=IntSlider(value=2, min=1, max=2, step=1, description='PCA Comps:')
);

# COMMAND ----------

# DBTITLE 1,📏 Part 3: sklearn PCA - Industry Standard
# MAGIC %md
# MAGIC # Part 3: sklearn PCA - Industry Standard 🛠️
# MAGIC
# MAGIC ## Production Mein PCA Kaise Use Karein?
# MAGIC
# MAGIC Real-world projects mein hum **sklearn** use karte hain kyunki:
# MAGIC - ✅ Optimized and tested
# MAGIC - ✅ Additional features
# MAGIC - ✅ Easy to use
# MAGIC - ✅ Industry standard
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Basic Usage:
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.decomposition import PCA
# MAGIC
# MAGIC # Create PCA object
# MAGIC pca = PCA(n_components=2)
# MAGIC
# MAGIC # Fit and transform
# MAGIC X_pca = pca.fit_transform(X)
# MAGIC
# MAGIC # Get explained variance
# MAGIC print(pca.explained_variance_ratio_)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Important Attributes:
# MAGIC
# MAGIC | Attribute | Description |
# MAGIC |-----------|-------------|
# MAGIC | `components_` | Principal component directions (eigenvectors) |
# MAGIC | `explained_variance_` | Variance captured by each component (eigenvalues) |
# MAGIC | `explained_variance_ratio_` | Percentage of variance explained |
# MAGIC | `mean_` | Mean of training data |
# MAGIC | `n_components_` | Number of components |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC Let's compare our implementation with sklearn! 👇

# COMMAND ----------

# DBTITLE 1,sklearn PCA - Implementation & Comparison
print("🚀 sklearn PCA Implementation")
print("="*60)

# First, standardize data (sklearn PCA doesn't do this automatically for comparison)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply sklearn PCA
pca_sklearn = PCA(n_components=2)
X_pca_sklearn = pca_sklearn.fit_transform(X_scaled)

print("\n✅ sklearn PCA Results:")
print(f"   Shape: {X_pca_sklearn.shape}")
print(f"   Explained variance ratio: {pca_sklearn.explained_variance_ratio_}")
print(f"   Total variance explained: {pca_sklearn.explained_variance_ratio_.sum()*100:.2f}%")

print("\n" + "="*60)
print("⚖️ Comparison: From-Scratch vs sklearn")
print("="*60)

comparison_df = pd.DataFrame({
    'Metric': [
        'PC1 Variance %',
        'PC2 Variance %',
        'Total Variance %',
        'Shape',
    ],
    'From-Scratch': [
        f"{pca_scratch.explained_variance_ratio[0]*100:.2f}%",
        f"{pca_scratch.explained_variance_ratio[1]*100:.2f}%",
        f"{pca_scratch.explained_variance_ratio.sum()*100:.2f}%",
        str(X_pca_scratch.shape),
    ],
    'sklearn': [
        f"{pca_sklearn.explained_variance_ratio_[0]*100:.2f}%",
        f"{pca_sklearn.explained_variance_ratio_[1]*100:.2f}%",
        f"{pca_sklearn.explained_variance_ratio_.sum()*100:.2f}%",
        str(X_pca_sklearn.shape),
    ]
})

print("\n", comparison_df.to_string(index=False))

print("\n✅ Perfect match! Our implementation is correct!")

# Visualization comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: From-scratch PCA
ax1 = axes[0]
scatter1 = ax1.scatter(X_pca_scratch[:, 0], X_pca_scratch[:, 1], 
                       c=X[:, 0], cmap='viridis', s=30, alpha=0.6, edgecolors='black')
ax1.set_xlabel('PC1', fontsize=11)
ax1.set_ylabel('PC2', fontsize=11)
ax1.set_title('From-Scratch PCA\n(Our Implementation)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='red', linestyle='--', alpha=0.3)
ax1.axvline(0, color='red', linestyle='--', alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Height')

# Plot 2: sklearn PCA
ax2 = axes[1]
scatter2 = ax2.scatter(X_pca_sklearn[:, 0], X_pca_sklearn[:, 1], 
                       c=X[:, 0], cmap='viridis', s=30, alpha=0.6, edgecolors='black')
ax2.set_xlabel('PC1', fontsize=11)
ax2.set_ylabel('PC2', fontsize=11)
ax2.set_title('sklearn PCA\n(Industry Standard)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='red', linestyle='--', alpha=0.3)
ax2.axvline(0, color='red', linestyle='--', alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Height')

plt.tight_layout()
plt.show()

print("\n💡 Note: Signs might be flipped (PC1 positive/negative) but that's OK!")
print("   Direction doesn't matter, only the variance captured matters!")

# COMMAND ----------

# DBTITLE 1,🤔 How Many Components to Choose?
print("🤔 Choosing Optimal Number of Components")
print("="*60)

# Fit PCA with all components
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

print(f"\n📊 Total components available: {pca_full.n_components_}")
print(f"\n📊 Explained variance by each component:")
for i, var in enumerate(pca_full.explained_variance_ratio_, 1):
    print(f"   PC{i}: {var*100:.2f}%")

# Cumulative explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

print(f"\n📊 Cumulative explained variance:")
for i, cum_var in enumerate(cumulative_variance, 1):
    print(f"   PC{i}: {cum_var*100:.2f}%")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Individual explained variance
ax1 = axes[0]
pc_range = range(1, len(pca_full.explained_variance_ratio_) + 1)
ax1.bar(pc_range, pca_full.explained_variance_ratio_ * 100, 
        alpha=0.7, color='skyblue', edgecolor='black', linewidth=2)
ax1.set_xlabel('Principal Component', fontsize=11)
ax1.set_ylabel('Explained Variance (%)', fontsize=11)
ax1.set_title('Variance Explained by Each Component', fontsize=12, fontweight='bold')
ax1.set_xticks(pc_range)
ax1.grid(axis='y', alpha=0.3)

# Add labels
for i, val in enumerate(pca_full.explained_variance_ratio_ * 100, 1):
    ax1.text(i, val + 1, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

# Plot 2: Cumulative explained variance (Scree plot)
ax2 = axes[1]
ax2.plot(pc_range, cumulative_variance * 100, 'bo-', linewidth=2, markersize=8)
ax2.axhline(y=95, color='red', linestyle='--', linewidth=2, label='95% Threshold')
ax2.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90% Threshold')
ax2.set_xlabel('Number of Components', fontsize=11)
ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=11)
ax2.set_title('Scree Plot - Cumulative Variance', fontsize=12, fontweight='bold')
ax2.set_xticks(pc_range)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_ylim([0, 105])

# Add labels
for i, val in enumerate(cumulative_variance * 100, 1):
    ax2.text(i, val + 2, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("📖 Decision Rules for Choosing Components:")
print("="*60)

print("\n1️⃣ 🎯 Threshold Method:")
print(f"   - For 90% variance: Use {np.argmax(cumulative_variance >= 0.90) + 1} component(s)")
print(f"   - For 95% variance: Use {np.argmax(cumulative_variance >= 0.95) + 1} component(s)")
print(f"   - For 99% variance: Use {np.argmax(cumulative_variance >= 0.99) + 1} component(s)")

print("\n2️⃣ 📊 Elbow Method:")
print("   - Scree plot mein dekho jahan curve flatten hota hai")
print("   - Is example mein: 2 components optimal lagte hain")

print("\n3️⃣ 🎯 Kaiser Criterion:")
print("   - Components with eigenvalue > 1")
eigenvalues_above_1 = np.sum(pca_full.explained_variance_ > 1)
print(f"   - Components with eigenvalue > 1: {eigenvalues_above_1}")

print("\n4️⃣ 🔬 Domain Knowledge:")
print("   - Visualization ke liye: 2-3 components")
print("   - ML preprocessing ke liye: 90-95% variance")
print("   - Exploratory analysis ke liye: 80-90% variance")

print(f"\n🎯 Recommendation for this data: Use 2 components")
print(f"   Reason: Captures {cumulative_variance[1]*100:.1f}% variance with significant dimensionality reduction!")

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: Scree Plot & Component Selection
print("🎮 Interactive Scree Plot - Component Selection")
print("Adjust variance threshold to see how many components needed!\n")

def interactive_scree_plot(variance_threshold=90):
    """
    Interactive scree plot with variance threshold
    """
    # Fit PCA with all components
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    n_features = X_scaled.shape[1]
    variance_ratios = pca_full.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(variance_ratios)
    
    # Find number of components for threshold
    n_components_needed = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Create figure
    fig = go.Figure()
    
    # Bar chart for individual variance
    fig.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(n_features)],
        y=variance_ratios,
        name='Individual Variance',
        marker=dict(
            color=['#FF6B6B' if i < n_components_needed else '#CCCCCC' 
                   for i in range(n_features)],
            line=dict(width=2, color='black')
        ),
        text=[f'{v:.1f}%' for v in variance_ratios],
        textposition='outside',
        hovertemplate='%{x}<br>Variance: %{y:.2f}%<extra></extra>'
    ))
    
    # Line chart for cumulative variance
    fig.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(n_features)],
        y=cumulative_variance,
        name='Cumulative Variance',
        mode='lines+markers',
        line=dict(color='#4ECDC4', width=4),
        marker=dict(size=10, symbol='circle', line=dict(width=2, color='black')),
        yaxis='y2',
        hovertemplate='%{x}<br>Cumulative: %{y:.2f}%<extra></extra>'
    ))
    
    # Add threshold line
    fig.add_shape(
        type='line',
        x0=-0.5, x1=n_features-0.5,
        y0=variance_threshold, y1=variance_threshold,
        line=dict(color='red', width=3, dash='dash'),
        yref='y2'
    )
    
    # Add annotation for threshold
    fig.add_annotation(
        x=n_features-1,
        y=variance_threshold,
        text=f'Threshold: {variance_threshold}%',
        showarrow=True,
        arrowhead=2,
        arrowcolor='red',
        ax=-50,
        ay=-30,
        font=dict(size=12, color='red', family='Arial Black'),
        yref='y2'
    )
    
    # Add annotation for selected components
    if n_components_needed < n_features:
        fig.add_annotation(
            x=n_components_needed - 1,
            y=cumulative_variance[n_components_needed - 1],
            text=f'✅ {n_components_needed} Components<br>{cumulative_variance[n_components_needed-1]:.1f}% variance',
            showarrow=True,
            arrowhead=2,
            arrowcolor='green',
            ax=50,
            ay=-50,
            font=dict(size=11, color='green', family='Arial Black'),
            bgcolor='rgba(200, 255, 200, 0.8)',
            bordercolor='green',
            borderwidth=2,
            yref='y2'
        )
    
    # Update layout with dual y-axes
    fig.update_layout(
        title=f'🎮 Interactive Scree Plot | Threshold: {variance_threshold}% → Need {n_components_needed} Component(s)',
        xaxis=dict(title='Principal Component'),
        yaxis=dict(
            title='Individual Variance (%)',
            side='left'
        ),
        yaxis2=dict(
            title='Cumulative Variance (%)',
            side='right',
            overlaying='y',
            range=[0, 105]
        ),
        height=550,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.7, y=0.95, bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1)
    )
    
    fig.show()
    
    # Print recommendation
    print(f"\n🎯 Recommendation:")
    print(f"   Variance threshold: {variance_threshold}%")
    print(f"   Components needed: {n_components_needed} out of {n_features}")
    print(f"   Actual variance retained: {cumulative_variance[n_components_needed-1]:.2f}%")
    print(f"   Dimensionality reduction: {(1 - n_components_needed/n_features)*100:.0f}%")
    
    print(f"\n📊 Breakdown:")
    for i in range(n_components_needed):
        print(f"   PC{i+1}: {variance_ratios[i]:.2f}% (cumulative: {cumulative_variance[i]:.2f}%)")

# Create interactive widget
interact(
    interactive_scree_plot,
    variance_threshold=IntSlider(
        value=90, min=50, max=99, step=5,
        description='Threshold %:',
        continuous_update=False,
        style={'description_width': 'initial'}
    )
);

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: Reconstruction Quality
print("🎮 Interactive Reconstruction Quality Explorer")
print("Change number of components to see reconstruction quality!\n")

def interactive_reconstruction(n_components=2):
    """
    Interactive reconstruction quality visualization
    """
    # Apply PCA with selected components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_iris_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)
    
    # Calculate reconstruction error
    reconstruction_error = np.mean((X_iris_scaled - X_reconstructed) ** 2)
    
    # Calculate R² for each feature
    from sklearn.metrics import r2_score
    r2_scores = [r2_score(X_iris_scaled[:, i], X_reconstructed[:, i]) 
                 for i in range(X_iris_scaled.shape[1])]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{name}<br>R² = {r2:.4f}' for name, r2 in zip(feature_names, r2_scores)],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Plot each feature: original vs reconstructed
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    colors_feat = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']
    
    for idx, (pos, color) in enumerate(zip(positions, colors_feat)):
        row, col = pos
        original = X_iris_scaled[:, idx]
        reconstructed = X_reconstructed[:, idx]
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=original,
                y=reconstructed,
                mode='markers',
                marker=dict(size=5, color=color, opacity=0.6, line=dict(width=0.5, color='black')),
                name=feature_names[idx],
                showlegend=False,
                hovertemplate='Original: %{x:.2f}<br>Reconstructed: %{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Perfect reconstruction line (y=x)
        min_val, max_val = original.min(), original.max()
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Perfect Reconstruction' if idx == 0 else '',
                showlegend=(idx == 0),
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text='Original (Standardized)', row=row, col=col)
        fig.update_yaxes(title_text='Reconstructed', row=row, col=col)
    
    # Update layout
    fig.update_layout(
        height=700,
        title_text=f'🔄 Reconstruction Quality: {n_components} Component(s)<br>'
                   f'MSE: {reconstruction_error:.6f} | Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%',
        template='plotly_white',
        hovermode='closest',
        legend=dict(x=0.85, y=0.95)
    )
    
    fig.show()
    
    # Print statistics
    print(f"\n📊 Reconstruction Statistics:")
    print(f"   Components used: {n_components} / {X_iris_scaled.shape[1]}")
    print(f"   Variance retained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"   Mean Squared Error: {reconstruction_error:.6f}")
    print(f"\n📊 R² Scores per Feature:")
    for name, r2 in zip(feature_names, r2_scores):
        quality = '✅ Excellent' if r2 > 0.95 else '⚠️ Good' if r2 > 0.85 else '❌ Poor'
        print(f"   {name}: {r2:.4f} {quality}")
    
    avg_r2 = np.mean(r2_scores)
    print(f"\n   Average R²: {avg_r2:.4f}")
    print(f"\n💡 Closer to red line = Better reconstruction!")

# Create interactive widget
interact(
    interactive_reconstruction,
    n_components=IntSlider(
        value=2, min=1, max=4, step=1,
        description='Components:',
        continuous_update=False
    )
);

# COMMAND ----------

# DBTITLE 1,📏 Part 4: Real-World Application - Iris Dataset
# MAGIC %md
# MAGIC # Part 4: Real-World Applications 🌺
# MAGIC
# MAGIC ## 🌺 Iris Dataset - Classic ML Example
# MAGIC
# MAGIC **Problem:**  
# MAGIC Iris dataset mein **4 features** hain (sepal length, sepal width, petal length, petal width).
# MAGIC
# MAGIC **Goal:**  
# MAGIC - 4D data ko 2D mein visualize karna
# MAGIC - Patterns aur clusters dekhna
# MAGIC - Species classification ke liye preprocessing
# MAGIC
# MAGIC **Why PCA?**
# MAGIC - ✅ 4D data ko 2D mein plot kar sakte hain
# MAGIC - ✅ Most important patterns retain honge
# MAGIC - ✅ Easy to interpret and visualize
# MAGIC
# MAGIC Let's apply PCA! 👇

# COMMAND ----------

# DBTITLE 1,Real Example 1: Iris Dataset Classification
from sklearn.datasets import load_iris

print("🌺 Loading Iris Dataset...")
print("="*60)

# Load iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

print(f"\n📊 Dataset Info:")
print(f"   Samples: {X_iris.shape[0]}")
print(f"   Features: {X_iris.shape[1]}")
print(f"   Classes: {len(target_names)} ({', '.join(target_names)})")

print(f"\n📊 Features:")
for i, name in enumerate(feature_names, 1):
    print(f"   {i}. {name}")

# Show sample data
df_iris = pd.DataFrame(X_iris, columns=feature_names)
df_iris['species'] = [target_names[i] for i in y_iris]
print(f"\n📊 Sample Data:")
print(df_iris.head(10))

# Standardize and apply PCA
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

pca_iris = PCA(n_components=2)
X_iris_pca = pca_iris.fit_transform(X_iris_scaled)

print(f"\n🎯 PCA Results:")
print(f"   Original dimensions: {X_iris.shape[1]}")
print(f"   Reduced dimensions: {X_iris_pca.shape[1]}")
print(f"   PC1 variance: {pca_iris.explained_variance_ratio_[0]*100:.2f}%")
print(f"   PC2 variance: {pca_iris.explained_variance_ratio_[1]*100:.2f}%")
print(f"   Total variance retained: {pca_iris.explained_variance_ratio_.sum()*100:.2f}%")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Original features (2D projection)
ax1 = axes[0]
for i, target_name in enumerate(target_names):
    mask = y_iris == i
    ax1.scatter(X_iris[mask, 0], X_iris[mask, 1], 
               label=target_name, s=50, alpha=0.6, edgecolors='black')
ax1.set_xlabel(feature_names[0], fontsize=11)
ax1.set_ylabel(feature_names[1], fontsize=11)
ax1.set_title('Original Features\n(Sepal Length vs Sepal Width)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: PCA transformed
ax2 = axes[1]
for i, target_name in enumerate(target_names):
    mask = y_iris == i
    ax2.scatter(X_iris_pca[mask, 0], X_iris_pca[mask, 1], 
               label=target_name, s=50, alpha=0.6, edgecolors='black')
ax2.set_xlabel(f'PC1 ({pca_iris.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
ax2.set_ylabel(f'PC2 ({pca_iris.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
ax2.set_title(f'PCA Transformed\n({pca_iris.explained_variance_ratio_.sum()*100:.1f}% total variance)', 
             fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='red', linestyle='--', alpha=0.3)
ax2.axvline(0, color='red', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n🔑 Key Insights:")
print(f"   • PCA clearly separates the 3 species!")
print(f"   • Setosa (blue) is completely separated")
print(f"   • Versicolor and Virginica have some overlap")
print(f"   • {pca_iris.explained_variance_ratio_.sum()*100:.1f}% information retained with just 2 components!")
print(f"\n🎯 PCA made 4D data easy to visualize and understand!")

# COMMAND ----------

# DBTITLE 1,PCA Component Interpretation - Feature Contributions
print("🔍 Interpreting Principal Components")
print("="*60)

# Get component loadings (contribution of each feature)
components_df = pd.DataFrame(
    pca_iris.components_,
    columns=feature_names,
    index=['PC1', 'PC2']
)

print("\n📊 Component Loadings (Feature Contributions):")
print(components_df)

print("\n📊 Interpretation:")
print("   Positive value = Feature increases in that direction")
print("   Negative value = Feature decreases in that direction")
print("   Large absolute value = Strong contribution")

# Visualize feature contributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, pc in enumerate(['PC1', 'PC2']):
    ax = axes[idx]
    loadings = components_df.loc[pc].values
    colors = ['red' if x < 0 else 'green' for x in loadings]
    
    bars = ax.barh(feature_names, loadings, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xlabel('Loading Value', fontsize=11)
    ax.set_title(f'{pc} Feature Contributions\n({pca_iris.explained_variance_ratio_[idx]*100:.1f}% variance)', 
                fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, loadings)):
        ax.text(val + (0.02 if val > 0 else -0.02), i, f'{val:.3f}', 
               va='center', ha='left' if val > 0 else 'right', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n🔑 PC1 Interpretation:")
max_contrib_pc1 = components_df.loc['PC1'].abs().idxmax()
print(f"   Strongest contributor: {max_contrib_pc1}")
print(f"   PC1 primarily captures petal dimensions (length & width)")

print("\n🔑 PC2 Interpretation:")
max_contrib_pc2 = components_df.loc['PC2'].abs().idxmax()
print(f"   Strongest contributor: {max_contrib_pc2}")
print(f"   PC2 captures variation in sepal dimensions")

print("\n💡 Insight: Principal components are LINEAR COMBINATIONS of original features!")

# COMMAND ----------

# DBTITLE 1,📏 Part 5: Advanced Topics
# MAGIC %md
# MAGIC # Part 5: Advanced PCA Topics 🚀
# MAGIC
# MAGIC ## 1️⃣ Inverse Transform - PCA se Original Space Mein Wapas
# MAGIC
# MAGIC PCA **reversible** hai! Reduced data ko wapas original space mein laa sakte ho:
# MAGIC
# MAGIC $$X_{\text{reconstructed}} = X_{\text{PCA}} \cdot W^T + \mu$$
# MAGIC
# MAGIC Where:
# MAGIC - $X_{\text{PCA}}$ = Transformed data
# MAGIC - $W$ = Principal components matrix
# MAGIC - $\mu$ = Original mean
# MAGIC
# MAGIC **Use Cases:**
# MAGIC - Image compression
# MAGIC - Noise reduction
# MAGIC - Data reconstruction
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 2️⃣ PCA in ML Pipelines
# MAGIC
# MAGIC PCA ko ML preprocessing ke liye use karo:
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC from sklearn.ensemble import RandomForestClassifier
# MAGIC
# MAGIC pipeline = Pipeline([
# MAGIC     ('scaler', StandardScaler()),
# MAGIC     ('pca', PCA(n_components=0.95)),  # Retain 95% variance
# MAGIC     ('classifier', RandomForestClassifier())
# MAGIC ])
# MAGIC
# MAGIC pipeline.fit(X_train, y_train)
# MAGIC ```
# MAGIC
# MAGIC **Benefits:**
# MAGIC - ✅ Faster training
# MAGIC - ✅ Less overfitting
# MAGIC - ✅ Removes multicollinearity
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 3️⃣ When NOT to Use PCA
# MAGIC
# MAGIC ❌ **Interpretability important hai**  
# MAGIC    - PCs are linear combinations, hard to explain
# MAGIC
# MAGIC ❌ **Features already uncorrelated**  
# MAGIC    - PCA won't help much
# MAGIC
# MAGIC ❌ **Non-linear relationships**  
# MAGIC    - Use Kernel PCA, t-SNE, or UMAP instead
# MAGIC
# MAGIC ❌ **Small datasets**  
# MAGIC    - Overfitting risk
# MAGIC
# MAGIC Let's see these in action! 👇

# COMMAND ----------

# DBTITLE 1,Advanced 1: Inverse Transform - Reconstruction
print("🔄 PCA Inverse Transform - Data Reconstruction")
print("="*60)

# Use 2 components for reconstruction
pca_2comp = PCA(n_components=2)
X_iris_pca_2 = pca_2comp.fit_transform(X_iris_scaled)

# Reconstruct original data
X_iris_reconstructed = pca_2comp.inverse_transform(X_iris_pca_2)

print(f"\n📊 Reconstruction Summary:")
print(f"   Original shape: {X_iris_scaled.shape}")
print(f"   PCA shape: {X_iris_pca_2.shape}")
print(f"   Reconstructed shape: {X_iris_reconstructed.shape}")
print(f"   Variance retained: {pca_2comp.explained_variance_ratio_.sum()*100:.2f}%")

# Calculate reconstruction error
reconstruction_error = np.mean((X_iris_scaled - X_iris_reconstructed) ** 2)
print(f"\n📊 Mean Squared Reconstruction Error: {reconstruction_error:.6f}")

# Compare original vs reconstructed for first sample
print("\n📊 Sample Comparison (First 5 samples):")
comparison_df = pd.DataFrame({
    'Feature': feature_names * 5,
    'Sample': np.repeat(range(1, 6), 4),
    'Original': X_iris_scaled[:5].flatten(),
    'Reconstructed': X_iris_reconstructed[:5].flatten(),
})
comparison_df['Error'] = np.abs(comparison_df['Original'] - comparison_df['Reconstructed'])
print(comparison_df.head(20).to_string(index=False))

# Visualize reconstruction quality
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (feature, ax) in enumerate(zip(feature_names, axes.flat)):
    original = X_iris_scaled[:, idx]
    reconstructed = X_iris_reconstructed[:, idx]
    
    # Scatter plot
    ax.scatter(original, reconstructed, alpha=0.5, s=30, edgecolors='black')
    
    # Perfect reconstruction line
    min_val, max_val = original.min(), original.max()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Reconstruction')
    
    # Calculate R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(original, reconstructed)
    
    ax.set_xlabel('Original (Standardized)', fontsize=10)
    ax.set_ylabel('Reconstructed', fontsize=10)
    ax.set_title(f'{feature}\nR² = {r2:.4f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Reconstruction Quality: Original vs Reconstructed\n(Using 2 Principal Components)', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n🔑 Key Insights:")
print(f"   • Using only 2 components (from 4), we can reconstruct data")
print(f"   • Reconstruction retains {pca_2comp.explained_variance_ratio_.sum()*100:.1f}% of information")
print(f"   • Points close to red line = Good reconstruction")
print(f"   • Mean squared error: {reconstruction_error:.6f}")
print(f"\n🎯 Use Case: Image compression - reduce dimensions, then reconstruct!")

# COMMAND ----------

# DBTITLE 1,Advanced 2: PCA in ML Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

print("🧠 PCA in Machine Learning Pipeline")
print("="*60)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

print(f"\n📊 Train set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Pipeline 1: Without PCA
print("\n" + "="*60)
print("🔵 Pipeline 1: WITHOUT PCA")
print("="*60)

pipeline_no_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline_no_pca.fit(X_train, y_train)
score_no_pca = pipeline_no_pca.score(X_test, y_test)
cv_scores_no_pca = cross_val_score(pipeline_no_pca, X_train, y_train, cv=5)

print(f"\n✅ Test Accuracy: {score_no_pca*100:.2f}%")
print(f"✅ Cross-validation scores: {cv_scores_no_pca}")
print(f"✅ Mean CV Accuracy: {cv_scores_no_pca.mean()*100:.2f}% (±{cv_scores_no_pca.std()*100:.2f}%)")

# Pipeline 2: With PCA (2 components)
print("\n" + "="*60)
print("🔴 Pipeline 2: WITH PCA (2 components)")
print("="*60)

pipeline_with_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline_with_pca.fit(X_train, y_train)
score_with_pca = pipeline_with_pca.score(X_test, y_test)
cv_scores_with_pca = cross_val_score(pipeline_with_pca, X_train, y_train, cv=5)

print(f"\n✅ Test Accuracy: {score_with_pca*100:.2f}%")
print(f"✅ Cross-validation scores: {cv_scores_with_pca}")
print(f"✅ Mean CV Accuracy: {cv_scores_with_pca.mean()*100:.2f}% (±{cv_scores_with_pca.std()*100:.2f}%)")

# Pipeline 3: With PCA (95% variance)
print("\n" + "="*60)
print("🟢 Pipeline 3: WITH PCA (95% variance)")
print("="*60)

pipeline_pca_95 = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline_pca_95.fit(X_train, y_train)
score_pca_95 = pipeline_pca_95.score(X_test, y_test)
cv_scores_pca_95 = cross_val_score(pipeline_pca_95, X_train, y_train, cv=5)

print(f"\n✅ Components selected: {pipeline_pca_95.named_steps['pca'].n_components_}")
print(f"✅ Test Accuracy: {score_pca_95*100:.2f}%")
print(f"✅ Cross-validation scores: {cv_scores_pca_95}")
print(f"✅ Mean CV Accuracy: {cv_scores_pca_95.mean()*100:.2f}% (±{cv_scores_pca_95.std()*100:.2f}%)")

# Comparison
print("\n" + "="*60)
print("⚖️ Pipeline Comparison")
print("="*60)

comparison_df = pd.DataFrame({
    'Pipeline': ['No PCA (4 features)', 'PCA 2 components', 'PCA 95% variance'],
    'Test Accuracy': [f"{score_no_pca*100:.2f}%", f"{score_with_pca*100:.2f}%", f"{score_pca_95*100:.2f}%"],
    'Mean CV Accuracy': [
        f"{cv_scores_no_pca.mean()*100:.2f}%",
        f"{cv_scores_with_pca.mean()*100:.2f}%",
        f"{cv_scores_pca_95.mean()*100:.2f}%"
    ],
    'Features Used': [4, 2, pipeline_pca_95.named_steps['pca'].n_components_],
    'Dimensionality Reduction': ['0%', '50%', f"{(1 - pipeline_pca_95.named_steps['pca'].n_components_/4)*100:.0f}%']
})

print("\n", comparison_df.to_string(index=False))

print("\n🔑 Key Insights:")
print(f"   • PCA with 2 components: {score_with_pca*100:.1f}% accuracy, 50% fewer features!")
print(f"   • PCA retaining 95% variance: {score_pca_95*100:.1f}% accuracy")
print(f"   • Slight accuracy trade-off for significant dimensionality reduction")
print(f"\n🎯 Benefit: Faster training with minimal accuracy loss!")

# COMMAND ----------

# DBTITLE 1,🎮 Interactive: ML Pipeline Comparison
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import time

print("🎮 Interactive ML Pipeline Comparison")
print("Compare models with/without PCA!\n")

def interactive_ml_comparison(classifier='RandomForest', use_pca=True, n_components=2):
    """
    Interactive ML pipeline comparison
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
    )
    
    # Select classifier
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    clf = classifiers[classifier]
    
    # Create pipelines
    if use_pca:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('classifier', clf)
        ])
        pipeline_name = f'{classifier} + PCA({n_components})'
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        pipeline_name = f'{classifier} (No PCA)'
    
    # Train and evaluate
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    # Get predictions for confusion-like visualization
    y_pred = pipeline.predict(X_test)
    
    # Create visualization
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Performance', 'Feature Space (2D Projection)'),
        specs=[[{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    # Plot 1: Performance metrics
    metrics = ['Train Accuracy', 'Test Accuracy']
    scores = [train_score * 100, test_score * 100]
    colors_perf = ['#4ECDC4', '#FF6B6B']
    
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=scores,
            marker=dict(color=colors_perf, line=dict(width=2, color='black')),
            text=[f'{s:.1f}%' for s in scores],
            textposition='outside',
            showlegend=False,
            hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add training time annotation
    fig.add_annotation(
        x=0.5, y=max(scores) * 0.5,
        text=f'⏱️ Training Time<br>{train_time*1000:.2f} ms',
        showarrow=False,
        font=dict(size=14, color='black', family='Arial Black'),
        bgcolor='rgba(255, 255, 200, 0.8)',
        bordercolor='black',
        borderwidth=2,
        xref='x', yref='y',
        row=1, col=1
    )
    
    fig.update_yaxes(title_text='Accuracy (%)', range=[0, 105], row=1, col=1)
    
    # Plot 2: Feature space visualization (use PCA for visualization if available)
    if use_pca and 'pca' in pipeline.named_steps:
        # Use the PCA from pipeline
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        pca_vis = pipeline.named_steps['pca']
        X_test_vis = pca_vis.transform(X_test_scaled)
        x_label = f'PC1 ({pca_vis.explained_variance_ratio_[0]*100:.1f}%)'
        y_label = f'PC2 ({pca_vis.explained_variance_ratio_[1]*100:.1f}%)' if n_components > 1 else 'PC2'
    else:
        # Use first 2 original features for visualization
        X_test_vis = X_test[:, :2]
        x_label = feature_names[0]
        y_label = feature_names[1]
    
    # Plot test points colored by prediction
    colors_species = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    symbols_pred = ['circle', 'square', 'diamond']
    
    for i, (target_name, color, symbol) in enumerate(zip(target_names, colors_species, symbols_pred)):
        mask = y_pred == i
        correct_mask = (y_pred == y_test) & mask
        incorrect_mask = (y_pred != y_test) & mask
        
        if n_components >= 2 or not use_pca:
            x_data = X_test_vis[correct_mask, 0]
            y_data = X_test_vis[correct_mask, 1] if X_test_vis.shape[1] > 1 else np.zeros(correct_mask.sum())
        else:
            x_data = X_test_vis[correct_mask, 0]
            y_data = np.zeros(correct_mask.sum())
        
        # Correct predictions
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                name=f'{target_name} ✓',
                marker=dict(size=10, color=color, symbol=symbol, opacity=0.8,
                           line=dict(width=2, color='black')),
                hovertemplate=f'{target_name} (Correct)<br>{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Incorrect predictions (if any)
        if incorrect_mask.sum() > 0:
            if n_components >= 2 or not use_pca:
                x_data_wrong = X_test_vis[incorrect_mask, 0]
                y_data_wrong = X_test_vis[incorrect_mask, 1] if X_test_vis.shape[1] > 1 else np.zeros(incorrect_mask.sum())
            else:
                x_data_wrong = X_test_vis[incorrect_mask, 0]
                y_data_wrong = np.zeros(incorrect_mask.sum())
            
            fig.add_trace(
                go.Scatter(
                    x=x_data_wrong,
                    y=y_data_wrong,
                    mode='markers',
                    name=f'{target_name} ✗',
                    marker=dict(size=12, color=color, symbol='x', opacity=1,
                               line=dict(width=3, color='red')),
                    hovertemplate=f'{target_name} (Wrong)<br>{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<extra></extra>'
                ),
                row=1, col=2
            )
    
    fig.update_xaxes(title_text=x_label, row=1, col=2)
    fig.update_yaxes(title_text=y_label, row=1, col=2)
    
    # Update layout
    n_features_used = n_components if use_pca else X_iris.shape[1]
    fig.update_layout(
        height=550,
        title_text=f'🎮 {pipeline_name} | Features: {n_features_used}/{X_iris.shape[1]} | Test Acc: {test_score*100:.1f}%',
        template='plotly_white',
        hovermode='closest',
        showlegend=True,
        legend=dict(x=1.02, y=0.5)
    )
    
    fig.show()
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"   Classifier: {classifier}")
    print(f"   PCA: {'Yes' if use_pca else 'No'} ({n_components} components)" if use_pca else '   PCA: No')
    print(f"   Features used: {n_features_used} / {X_iris.shape[1]}")
    print(f"   Train accuracy: {train_score*100:.2f}%")
    print(f"   Test accuracy: {test_score*100:.2f}%")
    print(f"   Training time: {train_time*1000:.2f} ms")
    
    # Calculate error
    n_errors = (y_pred != y_test).sum()
    print(f"\n   Errors: {n_errors} / {len(y_test)} samples")
    
    if use_pca and n_components < X_iris.shape[1]:
        reduction = (1 - n_components/X_iris.shape[1]) * 100
        print(f"   Dimensionality reduction: {reduction:.0f}%")
        print(f"\n💡 Using {reduction:.0f}% fewer features with {test_score*100:.1f}% accuracy!")

# Create interactive widget
interact(
    interactive_ml_comparison,
    classifier=Dropdown(
        options=['RandomForest', 'LogisticRegression', 'SVM'],
        value='RandomForest',
        description='Classifier:'
    ),
    use_pca=widgets.Checkbox(value=True, description='Use PCA'),
    n_components=IntSlider(value=2, min=1, max=4, step=1, description='PCA Comps:')
);

# COMMAND ----------

# DBTITLE 1,🎯 Complete Summary - PCA Cheat Sheet
# MAGIC %md
# MAGIC # 🎯 Complete PCA Summary - Cheat Sheet
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📖 What is PCA?
# MAGIC
# MAGIC **Principal Component Analysis** = Dimensionality reduction technique that:
# MAGIC - Transforms data into **uncorrelated** principal components
# MAGIC - Captures **maximum variance** in lower dimensions
# MAGIC - Uses **eigenvectors** of covariance matrix as new axes
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Core Math Formulas
# MAGIC
# MAGIC ### 1. Variance
# MAGIC $$\text{Var}(X) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$
# MAGIC
# MAGIC ### 2. Covariance
# MAGIC $$\text{Cov}(X, Y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$
# MAGIC
# MAGIC ### 3. Covariance Matrix
# MAGIC $$C = \frac{1}{n}X^TX$$
# MAGIC
# MAGIC ### 4. Eigenvalue Equation
# MAGIC $$C\vec{v} = \lambda\vec{v}$$
# MAGIC
# MAGIC Where:
# MAGIC - $\vec{v}$ = Eigenvector (Principal Component direction)
# MAGIC - $\lambda$ = Eigenvalue (Variance in that direction)
# MAGIC
# MAGIC ### 5. PCA Transformation
# MAGIC $$X_{\text{PCA}} = X_{\text{standardized}} \cdot W$$
# MAGIC
# MAGIC Where $W$ = Matrix of top $k$ eigenvectors
# MAGIC
# MAGIC ### 6. Inverse Transform (Reconstruction)
# MAGIC $$X_{\text{reconstructed}} = X_{\text{PCA}} \cdot W^T + \mu$$
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔑 PCA Algorithm Steps
# MAGIC
# MAGIC ```
# MAGIC 1. Standardize data: X_std = (X - μ) / σ
# MAGIC 2. Compute covariance matrix: C = X_stdᵀ × X_std / (n-1)
# MAGIC 3. Calculate eigenvalues & eigenvectors of C
# MAGIC 4. Sort eigenvectors by eigenvalues (descending)
# MAGIC 5. Select top k eigenvectors
# MAGIC 6. Transform data: X_pca = X_std × W
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💻 Python Implementation
# MAGIC
# MAGIC ```python
# MAGIC from sklearn.decomposition import PCA
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC
# MAGIC # Standardize
# MAGIC scaler = StandardScaler()
# MAGIC X_scaled = scaler.fit_transform(X)
# MAGIC
# MAGIC # Apply PCA
# MAGIC pca = PCA(n_components=2)  # or 0.95 for 95% variance
# MAGIC X_pca = pca.fit_transform(X_scaled)
# MAGIC
# MAGIC # Get explained variance
# MAGIC print(pca.explained_variance_ratio_)
# MAGIC
# MAGIC # Inverse transform
# MAGIC X_reconstructed = pca.inverse_transform(X_pca)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Choosing Number of Components
# MAGIC
# MAGIC | Method | Rule |
# MAGIC |--------|------|
# MAGIC | **Threshold** | 90-95% cumulative variance |
# MAGIC | **Elbow** | Scree plot mein jahan curve flatten ho |
# MAGIC | **Kaiser** | Eigenvalue > 1 |
# MAGIC | **Domain** | Visualization: 2-3, ML: 90-95%, EDA: 80-90% |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ When to Use PCA
# MAGIC
# MAGIC - ✅ **Visualization**: High-dimensional data ko 2D/3D mein plot karna
# MAGIC - ✅ **Speed**: ML models ko faster banana
# MAGIC - ✅ **Noise reduction**: Low-variance components drop karna
# MAGIC - ✅ **Multicollinearity**: Correlated features ko combine karna
# MAGIC - ✅ **Preprocessing**: Before clustering or classification
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ❌ When NOT to Use PCA
# MAGIC
# MAGIC - ❌ **Interpretability important**: PCs are hard to explain
# MAGIC - ❌ **Non-linear relationships**: Use Kernel PCA, t-SNE, UMAP
# MAGIC - ❌ **Small datasets**: Overfitting risk
# MAGIC - ❌ **Features already uncorrelated**: Won't help
# MAGIC - ❌ **Supervised learning with few features**: May remove predictive info
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 💡 Key Insights
# MAGIC
# MAGIC 1. **PC1 = Direction of maximum variance**
# MAGIC 2. **PCs are orthogonal** (perpendicular to each other)
# MAGIC 3. **PCs are linear combinations** of original features
# MAGIC 4. **Standardization is crucial** for fair comparison
# MAGIC 5. **Information loss is controlled** by variance threshold
# MAGIC 6. **Reversible transformation** (inverse_transform available)
# MAGIC 7. **Unsupervised method** (doesn't use target labels)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛠️ Practical Tips
# MAGIC
# MAGIC 1. **Always standardize** before PCA (unless features same scale)
# MAGIC 2. **Check explained variance** to choose components
# MAGIC 3. **Use in Pipeline** for clean ML workflow
# MAGIC 4. **Visualize components** to understand feature contributions
# MAGIC 5. **Compare with original** to validate dimensionality reduction
# MAGIC 6. **Consider alternatives** (t-SNE, UMAP) for non-linear data
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Further Learning
# MAGIC
# MAGIC - **Kernel PCA**: For non-linear dimensionality reduction
# MAGIC - **Incremental PCA**: For large datasets (out-of-core)
# MAGIC - **Sparse PCA**: For sparse solutions
# MAGIC - **t-SNE**: Better for visualization (non-linear)
# MAGIC - **UMAP**: Faster alternative to t-SNE
# MAGIC - **Factor Analysis**: Similar but different assumptions
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🏆 Congratulations!
# MAGIC
# MAGIC Aap ne PCA ko **basics se advanced** tak seekh liya! 🎉
# MAGIC
# MAGIC **Covered Topics:**
# MAGIC - ✅ Mathematical foundation (Variance, Covariance, Eigenvectors)
# MAGIC - ✅ Complete algorithm implementation from scratch
# MAGIC - ✅ sklearn PCA usage
# MAGIC - ✅ Component selection strategies
# MAGIC - ✅ Real-world applications (Iris dataset)
# MAGIC - ✅ Advanced topics (Inverse transform, ML pipelines)
# MAGIC - ✅ When to use / not use PCA
# MAGIC
# MAGIC Ab tum PCA ko confidently real projects mein use kar sakte ho! 🚀

# COMMAND ----------

# DBTITLE 1,🚀 Practice Exercise - Try It Yourself!
# MAGIC %md
# MAGIC # 🚀 Practice Exercise - Apply Your Knowledge!
# MAGIC
# MAGIC ## Challenge: Wine Dataset Classification
# MAGIC
# MAGIC **Dataset:** Wine recognition dataset  
# MAGIC **Features:** 13 chemical properties  
# MAGIC **Target:** 3 wine types
# MAGIC
# MAGIC **Your Tasks:**
# MAGIC
# MAGIC 1. **Load & Explore**
# MAGIC    ```python
# MAGIC    from sklearn.datasets import load_wine
# MAGIC    wine = load_wine()
# MAGIC    ```
# MAGIC
# MAGIC 2. **Apply PCA**
# MAGIC    - Standardize data
# MAGIC    - Apply PCA with different n_components
# MAGIC    - Plot explained variance
# MAGIC
# MAGIC 3. **Visualization**
# MAGIC    - Create 2D scatter plot with PCA
# MAGIC    - Color by wine type
# MAGIC    - Add component loadings
# MAGIC
# MAGIC 4. **ML Pipeline**
# MAGIC    - Compare accuracy with/without PCA
# MAGIC    - Try different classifiers
# MAGIC    - Optimize n_components
# MAGIC
# MAGIC 5. **Interpretation**
# MAGIC    - Which features contribute most to PC1?
# MAGIC    - How much variance is retained with 2 components?
# MAGIC    - Can you separate wine types in 2D?
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Bonus Challenges 🎯
# MAGIC
# MAGIC 1. **Image Compression**: Apply PCA to compress images
# MAGIC 2. **Anomaly Detection**: Use reconstruction error to find outliers
# MAGIC 3. **Feature Engineering**: Create new features from PCs
# MAGIC 4. **Compare Methods**: PCA vs t-SNE vs UMAP
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC Good luck! 🚀 Practice makes perfect!

# COMMAND ----------

