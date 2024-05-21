
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

df = pd.read_csv("Breast Cancer Wisconsin (Diagnostic) Data.csv")
# Select the features you want to use
selected_features = ['radius_worst', 'area_worst', 'concave points_mean', 'texture_worst']

# Separate features (X) and labels (y)
X = df[selected_features]
y = df['diagnosis']  # Assuming 'diagnosis' column contains the labels
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.fit_transform(X_test)
svm=SVC()
svm.fit(X_train,y_train)
joblib.dump(svm, 'svm_mnist_model.pkl')