import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set professional Seaborn style and context for presentation-ready visualization
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.9)

# Generate realistic synthetic data for customer engagement metrics
# Metrics: Purchase Frequency (times/month), Average Order Value ($), Customer Lifetime Value ($),
# Engagement Score (0-100), Retention Rate (%), NPS Score (0-100), Social Media Interactions (count/month),
# Email Open Rate (%)
np.random.seed(42)  # For reproducibility
n_samples = 1000

# Generate correlated data for realism (using a simple covariance structure)
metrics = ['Purchase_Frequency', 'Avg_Order_Value', 'CLV', 'Engagement_Score', 'Retention_Rate', 
           'NPS', 'Social_Interactions', 'Email_Open_Rate']

# Base random data
data = np.random.randn(n_samples, len(metrics))

# Introduce some positive correlations (e.g., higher purchase freq correlates with higher engagement)
# Simple adjustment: add correlations between related metrics
data[:, 1] += 0.5 * data[:, 0]  # Avg_Order_Value correlates with Purchase_Frequency
data[:, 2] += 0.7 * data[:, 1]  # CLV correlates with Avg_Order_Value
data[:, 3] += 0.6 * data[:, 0]  # Engagement_Score correlates with Purchase_Frequency
data[:, 4] += 0.8 * data[:, 3]  # Retention_Rate correlates with Engagement_Score
data[:, 5] += 0.5 * data[:, 4]  # NPS correlates with Retention_Rate
data[:, 6] += 0.4 * data[:, 3]  # Social_Interactions correlates with Engagement_Score
data[:, 7] += 0.3 * data[:, 3]  # Email_Open_Rate correlates with Engagement_Score

# Scale data to realistic ranges
data[:, 0] = np.abs(data[:, 0]) * 5 + 1  # Purchase_Frequency: 1-26
data[:, 1] = np.abs(data[:, 1]) * 50 + 20  # Avg_Order_Value: 20-270
data[:, 2] = np.abs(data[:, 2]) * 1000 + 100  # CLV: 100-10700
data[:, 3] = np.clip((data[:, 3] + 3) / 6 * 100, 0, 100)  # Engagement_Score: 0-100
data[:, 4] = np.clip((data[:, 4] + 2) / 4 * 100, 0, 100)  # Retention_Rate: 0-100
data[:, 5] = np.clip((data[:, 5] + 2) / 4 * 100, 0, 100)  # NPS: 0-100
data[:, 6] = np.abs(data[:, 6]) * 200 + 10  # Social_Interactions: 10-2010
data[:, 7] = np.clip((data[:, 7] + 1) / 2 * 100, 0, 100)  # Email_Open_Rate: 0-100

# Create DataFrame
df = pd.DataFrame(data, columns=metrics)

# Compute correlation matrix
corr = df.corr()

# Create the figure with specified size for 512x512 output at dpi=64
plt.figure(figsize=(8, 8))

# Generate the heatmap with professional styling
sns.heatmap(
    corr,
    annot=True,  # Add correlation values as annotations
    cmap='coolwarm',  # Professional color palette (red positive, blue negative)
    center=0,  # Center the colormap at 0 for better contrast
    square=True,  # Make cells square
    linewidths=0.5,  # Add grid lines for clarity
    cbar_kws={'label': 'Correlation Strength'}  # Customize colorbar label
)

# Add title and axis labels
plt.title('Customer Engagement Metrics Correlation Matrix\n(Stamm LLC Analytics)', fontsize=16, pad=20)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Metrics', fontsize=12)

# Tight layout to prevent clipping
plt.tight_layout()

# Save as PNG with exact 512x512 dimensions (8x8 at 64 dpi)
plt.savefig('chart.png', dpi=64, bbox_inches='tight', facecolor='white')

# Close the figure to free memory
plt.close()