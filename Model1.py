import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pickle

def integration_by_step(y0, f_y, y_min, y_max, rho=5515.0, step=100, dir=1, grav=None):
    """
        This function integrates a one dimensional differential equation by finite sum following Euler method.

        Parameters
        ------
        y0 : float
            Boundary condition
        f_y : function
            ODE to be solved
        y_min : float
            Minimum radius, lower limit of integration
        y_max : float
            Maximum radius, upper limit of integration
        rho : float
            Mean density for the layer
        step : float, optional
            finite sum integration step
            (default = 1 m)
        dir : int, optional
            direction of the integration (upward if 1, downward if 0)
            (default = 1)
        grav : numpy array, optional
            gravity distribution by radius, used in downward integration

        Returns
        -----
        y : numpy array
            The integrated values of y = f_y
        r : numpy array
            The corresponding radius of the integrated quantity

    """
    iterations = math.ceil((y_max - y_min) / step)
    y = np.zeros(iterations + 1)
    r = np.zeros(iterations + 1)
    y[0] = y0

    if dir:
        r[0] = y_min
        for i in range(1, iterations + 1):
            r[i] = r[i - 1] + step
            y[i] = y[i - 1] + f_y(r[i], rho) * step

    else:
        r[0] = y_max
        for i in range(1, iterations + 1):
            r[i] = r[i - 1] - step
            y[i] = y[i - 1] + f_y(r[i], rho, grav) * step

    return y, r


def dM_dr(r, rho):
    """

    :param r: float. Radius from center of the planet [m]
    :param rho: float. Average density of the considered layer [kg/m^3]
    :return: mass distribution [kg]
    """
    return 4 * np.pi * rho * r ** 2


def dp_dr(r, rho, gravity, step=100):
    """

    :param r: float. Radius from center of the planet [m]
    :param rho: float. Average density of the considered layer [kg/m^3]
    :param gravity: numpy array. Gravity distribution [m/s^2]
    :param step: float. Step size [m]
    :return: pressure distribution [Pa]
    """
    rad = math.ceil(r / step)
    return rho * gravity[rad]


path = './output/model1'
if not os.path.exists(path):
    os.makedirs(path)


G = 6.6743e-11  # Gravitational constant m^3/kg/s^2
p0 = 0  # Atmospheric pressure
step = 100
verify_code = 0  # 1 if testing the code, 0 otherwise
user_input = 0  # 1 if acquiring user's input, 0 otherwise

if verify_code:
    # Earth constant density
    R = np.array([0, 6371e3])
    rho = np.array([5515])
elif user_input:
    planet_name = input('What planet are you modelling? ')
    n_layers = int(input('How many layers do you want? '))
    R = np.zeros(n_layers + 1)
    rho = np.zeros(n_layers)
    print('Insert the radius of the most internal layer (or the planet radius if only one layer is selected '
          'and its average density')
    R[1] = float(input('Radius [m]: '))
    R[1] = (R[1]//step)*step
    rho[0] = float(input('Mean density [kg/m^3]: '))
    current_layer = 1
    while current_layer < n_layers:
        print('Insert radius (from planet\'s center) and average density of next layer')
        R[current_layer + 1] = float(input('Radius [m]: '))
        R[current_layer + 1] = (R[current_layer + 1]//step)*step
        rho[current_layer] = float(input('Mean density [kg/m^3]: '))
        current_layer += 1
    print('Input assigned, starting computation')
else:
    planet_name = 'Mars'
    '''
    # Samuel
    R = np.array([0, 1.7e6, 1.7e6+400e3, 3.2895e6, 3.3895e6])
    rho = np.array([6650, 4250, 3500, 2900])
    '''
    # Steinberger
    R = np.array([0, 1.6385e6, 3.3895e6-240e3, 3.3895e6-50e3, 3.3895e6])
    rho = np.array([7300, 3600, 3362, 2950])
    '''
    # Earth verification Giordano
    R = np.array([0, 3110e3, 3110e3+2911e3, 3110e3+2911e3+30e3])
    rho = np.array([10500, 4000, 2800])
    '''

# preallocate variables
mass = np.array([])
radius_m = np.array([])
radius_p = np.array([])
gravity = np.array([])
pressure = np.array([])
y0 = 0

for i in range(len(rho)):
    # Define mass and gravity through upwards integration
    y_min = R[i]
    y_max = R[i + 1]
    rho_new = rho[i]
    mass_new, radius_m_new = integration_by_step(y0, dM_dr, y_min, y_max, rho=rho_new)
    y0 = mass_new[-1]
    gravity_new = mass_new * G / radius_m_new ** 2
    mass = np.append(mass, mass_new)
    radius_m = np.append(radius_m, radius_m_new)
    gravity = np.append(gravity, gravity_new)

y0 = p0
n = len(R)
for i in range(len(rho)):
    # Define density through downwards integration
    y_min = R[n - i - 2]
    y_max = R[n - i - 1]
    rho_new = rho[n - i - 2]
    pressure_new, radius_p_new = integration_by_step(y0, dp_dr, y_min, y_max, rho=rho_new, dir=0, grav=gravity)
    y0 = pressure_new[-1]
    radius_p = np.append(radius_p, radius_p_new)
    pressure = np.append(pressure, pressure_new)

print('Total mass: ', mass[-1], '[kg]')

# mean density
rho_mean = mass[-1] / (4 / 3 * np.pi * radius_m[-1] ** 3)
print('Mean density: ', rho_mean, '[kg/m^3]')

# evaluate moment of inertia
I = 0
for i in range(len(rho)):
    # Define moment of inertia for spherical shell
    index_1 = np.where(radius_m == R[i])[0][0]
    index_2 = np.where(radius_m == R[i + 1])[0][0]
    I = I + 2 / 5 * (mass[index_2] - mass[index_1]) * (R[i + 1] ** 5 - R[i] ** 5) / (R[i + 1] ** 3 - R[i] ** 3)

print('Moment of inertia normalized :', I / (mass[-1] * radius_m[-1] ** 2))

path_results = f'{path}/results.txt'
with open(path_results, "w+") as file:
    file.write('NUMERICAL RESULTS\n')
    file.write(f'Total mass is {mass[-1]:.6g} kg\n')
    file.write(f'Total mean density is {rho_mean:.6g} kg/m^3\n')
    file.write(f'Normalized MoI is {I / (mass[-1] * radius_m[-1] ** 2):.4g}\n')

if verify_code:
    mass_th = 4 / 3 * np.pi * R[-1] ** 3 * rho[0]
    P_th = 3 * G * mass[-1] ** 2 / (8 * np.pi * R[-1] ** 4)
    print('Code verification')
    print('Theoretical mass: ', mass_th, '[kg]')
    print('Numerical mass: ', mass[-1], '[kg]')
    print('Difference: ', mass_th - mass[-1], '[kg]\nDifference (percentage): ', (mass_th - mass[-1]) / mass_th * 100,
          '%')
    print('-------------------------')
    print('Theoretical core pressure: ', P_th, '[Pa]')
    print('Numerical core pressure: ', pressure[-2], '[Pa]')
    print('Difference: ', P_th - pressure[-2], '[Pa]\nDifference (percentage): ', (P_th - pressure[-2]) / P_th * 100,
          '%')

data = {
    'mass': mass,
    'gravity': gravity,
    'pressure': pressure,
    'radius': radius_m,
    'mean density': rho_mean,
    'Normalized MoI': I / (mass[-1] * radius_m[-1] ** 2)
}


path_data = f'{path}/data.pkl'
with open(path_data, 'wb') as f:
    pickle.dump(data, f)


path_plot = f'{path}/plot'
if not os.path.exists(path_plot):
    os.makedirs(path_plot)


lw = 2
plt.figure(figsize=(14, 8))
plt.rc('font', size=18)
if verify_code:
    plt.suptitle('Homogeneous density planet')
else:
    plt.suptitle(f'{planet_name} layered model')
plt.subplot(1, 3, 1)
plt.plot(mass / 1e23, radius_m / 1e3, lw=lw)
plt.xlabel('Integrated mass ($10^{23}$ Kg)')
plt.ylabel('Radius (km)')
plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
plt.subplot(1, 3, 2)
plt.plot(gravity, radius_m / 1e3, lw=lw)
plt.xlabel('Gravity (m/$s^2$)')
plt.ylabel('Radius (km)')
plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
plt.subplot(1, 3, 3)
plt.plot(pressure / 1e9, radius_p / 1e3, lw=lw)
plt.xlabel('Pressure (GPa)')
plt.ylabel('Radius (km)')
plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(f'{path_plot}/layered_model.png')
plt.show()

if user_input or not verify_code:
    print('Computation is ended, results are saved in:', path_results, '\nplot are saved in:', path_plot,
          '\ndata are saved in:', path_data)
