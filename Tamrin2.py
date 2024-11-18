import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
data =pd.read_csv('spg.csv')
X = data[['snowfall_amount_sfc', 'wind_speed_10_m_above_gnd', 'wind_direction_10_m_above_gnd', 'wind_speed_80_m_above_gnd', 'wind_direction_80_m_above_gnd', 'wind_speed_900_mb', 'wind_direction_900_mb', 'wind_gust_10_m_above_gnd']]
y = data['temperature_2_m_above_gnd']
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred= model.predict(X_test)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 =r2_score(y_test,y_pred)
print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2Score:{r2}")
plt.figure(figsize=(14,7))
y_test_sorted= y_test.sort_index()
y_pred_sorted = pd.Series(y_pred,index=y_test_sorted.index)
plt.plot(y_test_sorted.index,y_test_sorted.values, label="Actual", color='yellow')
plt.plot(y_pred_sorted.index,y_pred_sorted.values, label="Predicted", color='black',linestyle='--')
plt.xlabel("DateTime")
plt.ylabel("Global Active Power (kW)")
plt.title("Actual vs Predicted")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()