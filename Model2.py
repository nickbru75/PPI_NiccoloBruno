import numpy as np
import matplotlib.pyplot as plt


def conductive_layer(T0, grad, z):
    T = np.zeros(shape=z.shape)
    T = T0 + grad * z
    return T


def convective_layer(T_avg, z, alpha, grav, cp):
    T = np.zeros(shape=z.shape)
    T[0] = T_avg
    const = alpha * grav * T_avg / cp
    for i in range(1, len(T)):
        T[i] = T[i-1] + const * (z[i] - z[i - 1])
    return T


def density(rho_0, alpha_T, k, T, p):
    rho = np.zeros(shape=T.shape)
    rho[0] = rho_0
    T = T[::-1]
    p = p[::-1]
    for i in range(1, len(rho)):
        # print(alpha_T * (T[i] - T[0]), 1/k * (p[i] - p[0]))
        rho[i] = rho_0 * (1 - alpha_T * (T[i] - T[0]) + 1/k * (p[i] - p[0]))
    return rho


step = 100
G = 6.6743e-11      # Gravitational constant m^3/kg/s^2
grad = 0.0003
T0_in = 288.15
T_avg = 288
alpha = 3e-5
alpha_T = alpha
cp = 1200
K = 130e9

R = np.array([0, 1.675e6, 3.2895e6, 3.3895e6])
radius = np.arange(0, R[-1]+step, step)
rho_layer = np.array([7200, 3500, 2900])
convective = np.array([0, 1, 0])

index_old = 0
rho = np.ones(shape=radius.shape)
for i in range(len(rho_layer)):
    index = np.where(radius == R[i + 1])[0][0]
    rho[index_old:index + 1] *= rho_layer[i]
    index_old = index

tol = 0.1
rho_old = rho + tol + 1
n_iter = 0

while np.average(np.abs(rho - rho_old)) > tol:
    rho_old = rho
    mass = np.zeros(shape=radius.shape)
    gravity = np.zeros(shape=radius.shape)
    j = 1
    i = 1
    while i < len(mass):
        while j < len(R) and radius[i] < R[j]:
            mass[i] = mass[i-1] + 4*np.pi*rho[i]*radius[i]**2*step
            gravity[i] = G * mass[i] / radius[i] ** 2
            i += 1
        mass[i] = mass[i-1]
        gravity[i] = G * mass[i] / radius[i] ** 2
        i += 1
        j += 1

    pressure = np.zeros(shape=radius.shape)
    i = 1
    j = 2
    while i < len(pressure):
        while j <= len(R) and radius[-i] > R[-j]:
            pressure[i] = pressure[i - 1] + rho[-i] * gravity[-i] * step
            i += 1

        if i != len(pressure):
            pressure[i] = pressure[i-1]
        i += 1
        j += 1
    pressure = pressure[::-1]

    index_old = len(radius)
    temp = np.array([])
    rho = np.array([])
    T0 = T0_in
    for i in range(1, len(rho_layer)+1):
        rho0 = rho_layer[-i]
        index = np.where(radius == R[-i-1])[0][0]
        z = -(radius[index:index_old] - R[-i])[::-1]
        pr = pressure[index:index_old]
        grav = np.average(gravity[index:index_old])
        index_old = index
        if convective[-i] == 0:
            T = conductive_layer(T0, grad, z)
            T_avg = T[-1]
            temp = np.append(temp, T)
            T = T[::-1]
            rho = np.append(rho, density(rho0, alpha_T, K, T, pr))
        else:
            T = convective_layer(T_avg, z, alpha, grav, cp)
            T0 = T[-1]
            temp = np.append(temp, T)
            T = T[::-1]
            rho = np.append(rho, density(rho0, alpha_T, K, T, pr))
    temp = temp[::-1]
    n_iter += 1
    rho = rho[::-1]
print(n_iter)

print('Mass :', mass[-1], 'kg')
I = 8*np.pi/3*np.sum(rho*radius**4*step)
print('Moment of inertia normalized :', I/(mass[-1]*radius[-1]**2))

plt.figure()
plt.plot(rho, radius)
plt.plot(temp, radius)
plt.show()


lw = 2
plt.figure(figsize=(14, 8))
plt.rc('font', size=18)
plt.subplot(1, 3, 1)
plt.plot(mass / 1e23, radius / 1e3, lw=lw)
plt.xlabel('Integrated mass ($10^{23}$ Kg)')
plt.ylabel('Radius (km)')
plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
plt.subplot(1, 3, 2)
plt.plot(gravity, radius / 1e3, lw=lw)
plt.xlabel('Gravity (m/$s^2$)')
plt.ylabel('Radius (km)')
plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
plt.subplot(1, 3, 3)
plt.plot(pressure / 1e9, radius / 1e3, lw=lw)
plt.xlabel('Pressure (GPa)')
plt.ylabel('Radius (km)')
plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
plt.tight_layout()
plt.show()
