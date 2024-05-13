import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

path = './Result reproducibility'

filename = [f'{path}/data_model1_4layers.pkl', f'{path}/data_model1_enrichedlayer.pkl',
            f'{path}/data_model2.pkl', f'{path}/data_model3.pkl']

with open(filename[0], 'rb') as f:
    data = pickle.load(f)
radius_1 = data['radius']
gravity_1 = data['gravity']
pressure_1 = data['pressure']
radius_density_1 = np.array([0, 1.6385e6, 1.6385e6, 3.3895e6-240e3, 3.3895e6-240e3, 3.3895e6-50e3, 3.3895e6-50e3, 3.3895e6])
density_1 = np.array([7300, 7300, 3600, 3600, 3362, 3362, 1950, 2950])

with open(filename[1], 'rb') as f:
    data = pickle.load(f)
radius_2 = data['radius']
gravity_2 = data['gravity']
pressure_2 = data['pressure']
radius_density_2 = np.array([0, 1.7e6, 1.7e6, 1.7e6 + 400e3, 1.7e6 + 400e3, 3.2895e6, 3.2895e6, 3.3895e6])
density_2 = np.array([6650, 6650, 4250, 4250, 3500, 3500, 2900, 2900])

with open(filename[2], 'rb') as f:
    data = pickle.load(f)
data = data[2100]
radius_3 = data['radius']
gravity_3 = data['gravity']
pressure_3 = data['pressure']
temperature_3 = data['temperature']
density_3 = data['density']

with open(filename[3], 'rb') as f:
    data = pickle.load(f)
radius_4 = data['radius']
gravity_4 = data['gravity']
pressure_4 = data['pressure']
temperature_4 = data['temperature']
density_4 = data['density']


lw = 2
plt.figure(figsize=(12, 12))
plt.rc('font', size=18)
plt.subplot(2, 2, 1)
plt.plot(gravity_1, radius_1/1e3, linewidth=lw, label='Model 1')
plt.plot(gravity_2, radius_2/1e3, linewidth=lw, label='Model 1 extra layer')
plt.plot(gravity_3, radius_3/1e3, linewidth=lw, label='Model 2 (average)')
plt.plot(gravity_4, radius_4/1e3, linewidth=lw, label='Model 3')
plt.xlabel('Gravity (m/$s^2$)')
plt.ylabel('Radius (Km)')
plt.legend(fontsize=14)

plt.subplot(2, 2, 2)
plt.plot(pressure_1[::-1]/1e9, radius_1/1e3, linewidth=lw, label='Model 1')
plt.plot(pressure_2[::-1]/1e9, radius_2/1e3, linewidth=lw, label='Model 1 extra layer')
plt.plot(pressure_3/1e9, radius_3/1e3, linewidth=lw, label='Model 2 (average)')
plt.plot(pressure_4/1e9, radius_4/1e3, linewidth=lw, label='Model 3')
plt.xlabel('Pressure (GPa)')
plt.ylabel('Radius (Km)')
plt.legend(fontsize=14)

ax1 = plt.subplot(2, 2, 3)
plt.plot(density_1, radius_density_1/1e3, linewidth=lw, label='Model 1')
plt.plot(density_2, radius_density_2/1e3, linewidth=lw, label='Model 1 extra layer')
plt.plot(density_3, radius_3/1e3, linewidth=lw, label='Model 2 (average)')
plt.plot(density_4, radius_4/1e3, linewidth=lw, label='Model 3')
plt.xlabel('Density (Kg/$m^3$)')
plt.ylabel('Radius (Km)')
plt.legend(fontsize=14)

plt.tight_layout()
plt.show()