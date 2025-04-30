import pandas as pd
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('final_data.csv')
data.columns = data.columns.str.strip() # remove white space
#breakpoint()

# histogram
plt.hist(data['mean_temp'], bins=10, edgecolor='black')
plt.title('Distribution of Mean Temperatures')
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('Frequency')


plot_filename = 'mean_temp_distribution.png'
plt.savefig(plot_filename)
plt.close()

# metrics
avg_mean_temp = data['mean_temp'].mean()

avg_std_temp = data['std_temp'].mean()

buildings_above_18 = data[data['pct_above_18'] >= 0.50].shape[0]

buildings_below_15 = data[data['pct_below_15'] >= 0.50].shape[0]

# save
results = (
    f"Average Mean Temperature of the Buildings: {avg_mean_temp:.2f}°C\n"
    f"Average Temperature Standard Deviation: {avg_std_temp:.2f}°C\n"
    f"Number of Buildings with at least 50% of Area Above 18ºC: {buildings_above_18}\n"
    f"Number of Buildings with at least 50% of Area Below 15ºC: {buildings_below_15}\n"
)


txt_filename = 'building_temperature_analysis.txt'
with open(txt_filename, 'w') as file:
    file.write(results)

print(f"Results have been saved to: {txt_filename}")
print(f"Plot has been saved to: {plot_filename}")
