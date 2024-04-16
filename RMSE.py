import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read data from CSV file
df = pd.read_csv("duzenlenmis_otel_veritabani_DB3.csv")

# Drop observations with missing values from the dataset
df.dropna(inplace=True)

# Create training and test sets
X = df.drop(['otel_ad', 'fiyat', 'il', 'ilce'], axis=1) # Independent variables
y = df['fiyat'] # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate error metrics
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

print("Training set RMSE:", train_rmse)
print("Test set RMSE:", test_rmse)

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
