import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('Japan_processed.csv')
print(df.head())

# Checking for missing values
missing_per_column = df.isnull().sum()
print(missing_per_column)

fig, ax = plt.subplots(figsize=(7, 5))

# Plot Magnitude
#---------------
ax.plot(df.index, df['Magnitude'], color='b')
ax.set_ylabel('Magnitude')
ax.set_xlabel('Index of Earthquake in Database')
ax.set_title('Magnitude of Earthquakes in Japan')
# Set y-ticks for Magnitude from 3.2 to 6.0 with a step of 0.2
magnitude_ticks = np.arange(3.0, 6.25, 0.25)
ax.set_yticks(magnitude_ticks)
plt.show()          # Display the plot


# Plotting average mag per month and year
#-----------------------------------------
# Group the data by month and year and calculate the mean and median for mag and magError
monthly_stats = df.groupby('Month')[['Magnitude']].agg(['mean'])
yearly_stats = df.groupby('Year')[['Magnitude']].agg(['mean'])
plt.figure(figsize=(8, 5))

# First plot: Average mag per month
plt.subplot(1, 2, 1)
plt.plot(monthly_stats.index, monthly_stats['Magnitude']['mean'], label='Avg Magnitude (Mean)', marker='o')
x_ticks = np.arange(1, 12, 1)
plt.xticks(ticks = x_ticks)
plt.title('Average Magnitude per Month')
plt.xlabel('Month')
plt.ylabel('Avg Magnitude')
plt.legend()

# Second plot: Average mag per year with mean and median
plt.subplot(1, 2, 2)
plt.plot(yearly_stats.index, yearly_stats['Magnitude']['mean'], label='Avg Magnitude (Mean)', marker='o')
x_ticks = np.arange(2019, 2024, 1)
plt.xticks(ticks = x_ticks)
plt.title('Average Magnitude per Year')
plt.xlabel('Year')
plt.ylabel('Avg Magnitude')
plt.legend()

plt.tight_layout()
plt.show()

# Plotting earthquake locations
#------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df['Longitude'], df['Latitude'], c=df['Magnitude'], cmap='coolwarm', s=20)
plt.colorbar(label='Magnitude')
plt.title('Earthquake Locations (Color-coded by Magnitude)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()