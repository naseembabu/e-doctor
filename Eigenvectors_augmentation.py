import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dia_df = pd.read_csv("/content/zero_pca - Sheet1.csv")
# separating the data and labels
X = dia_df.drop(columns = ['status'], axis=1)
y = dia_df['status']

# Convert X to a DataFrame if it's a NumPy array
if not isinstance(X, pd.DataFrame):
    X = pd.DataFrame(X)

# Apply PCA
pca = PCA(n_components=4)  # Specify the number of principal components
X_pca = pca.fit_transform(X)

# Inverse PCA to reconstruct the data
X_reconstructed = pca.inverse_transform(X_pca)

# Convert the reconstructed data back to a DataFrame
reconstructed_df = pd.DataFrame(X_reconstructed, columns=X.columns)

# Save the reconstructed DataFrame to an Excel sheet
reconstructed_df.to_excel('dia_pca_04.xlsx', index=False)
