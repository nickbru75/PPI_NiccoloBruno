import numpy as np
import burnman
from burnman import Composition
from burnman.tools.chemistry import formula_mass
from burnman import BoundaryLayerPerturbation
import matplotlib.pyplot as plt
from burnman import Mineral, PerplexMaterial, Composite, Layer, Planet
from burnman import minerals
import pickle
import os


path = './output/model3'
if not os.path.exists(path):
    os.makedirs(path)

# Composition of the Core of Mars
core_composition = Composition({'Fe': 0.75, 'S': 0.25}, 'weight')

for c in [core_composition]:
    c.renormalize('atomic', 'total', 1.)

core_elemental_composition = dict(core_composition.atomic_composition)
core_molar_mass = formula_mass(core_elemental_composition)

CMB_radius = 1638.5e3
core = Layer('core', radii=np.linspace(0., CMB_radius, 21))
hcp_iron = minerals.SE_2015.hcp_iron()
params = hcp_iron.params

params['name'] = 'modified liquid iron'
params['formula'] = core_elemental_composition
params['molar_mass'] = core_molar_mass
delta_V = 2.0e-7

core_material = Mineral(params=params,
                        property_modifiers=[['linear',
                                             {'delta_E': 0.,
                                              'delta_S': 0.,
                                              'delta_V': delta_V}]])

core.set_material(core_material)
core.set_temperature_mode('adiabatic')
core.set_pressure_mode(pressure_mode='self-consistent',
                       pressure_top=20.e9,
                       gravity_bottom=0.)

# MANTLE
lab_radius = 3289.e3
lab_temperature = 2100.

convecting_mantle_radii = np.linspace(CMB_radius, lab_radius, 101)
convecting_mantle = Layer('convecting mantle', radii=convecting_mantle_radii)

rock = Composite([minerals.SLB_2022.olivine(molar_fractions=[0.7, 0.3]),
                  minerals.SLB_2022.clinopyroxene(molar_fractions=[0., 0.5, 0.5, 0., 0.]),
                  minerals.SLB_2022.mgpv(),
                  minerals.SLB_2022.fepv()],
                 [0.4, 0.4, 0.1, 0.1])

convecting_mantle.set_material(rock)
convecting_mantle.set_temperature_mode('adiabatic', temperature_top=lab_temperature)

# Lithosphere
planet_radius = 3389.5e3
surface_temperature = 220.
rock = Composite([minerals.SLB_2022.olivine(molar_fractions=[0.6, 0.4]),
                  minerals.SLB_2022.clinopyroxene(molar_fractions=[0., 0.5, 0.5, 0., 0.]),
                  minerals.SLB_2022.orthopyroxene(molar_fractions=[0.475, 0.475, 0., 0.05]),
                  minerals.SLB_2022.plagioclase(molar_fractions=[0.4, 0.6]),
                  minerals.SLB_2022.mag(),
                  minerals.SLB_2022.ilmenite(molar_fractions=[0.5, 0.5, 0.])],
                 [0.1, 0.2, 0.2, 0.4, 0.05, 0.05])

andesine = minerals.SLB_2011.plagioclase(molar_fractions=[0.4, 0.6])
crust = Layer('crust', radii=np.linspace(lab_radius, planet_radius, 11))
crust.set_material(rock)
crust.set_temperature_mode('user-defined',
                           np.linspace(lab_temperature,
                                       surface_temperature, 11))

planet_mars = Planet('Planet Mars',
                     [core,
                      convecting_mantle,
                      crust], verbose=True)
planet_mars.make()

path_plot = f'{path}/plot'
if not os.path.exists(path_plot):
    os.makedirs(path_plot)

# PLOT
fig = plt.figure(figsize=(8, 5))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

bounds = np.array([[layer.radii[0] / 1.e3, layer.radii[-1] / 1.e3]
                   for layer in planet_mars.layers])
maxy = [15, 400, 12, 7000]
for bound in bounds:
    for i in range(4):
        ax[i].fill_betweenx([0., maxy[i]],
                            [bound[0], bound[0]],
                            [bound[1], bound[1]], alpha=0.2)

ax[0].plot(planet_mars.radii / 1.e3, planet_mars.density / 1.e3,
           label=planet_mars.name)
ax[0].set_ylabel('Density ($10^3$ kg/m$^3$)')
ax[0].set_ylim([0, 8])

# Make a subplot showing the calculated pressure profile
ax[1].plot(planet_mars.radii / 1.e3, planet_mars.pressure / 1.e9)
ax[1].set_ylabel('Pressure (GPa)')
ax[1].set_ylim([0, 50])

# Make a subplot showing the calculated gravity profile
ax[2].plot(planet_mars.radii / 1.e3, planet_mars.gravity)
ax[2].set_ylabel('Gravity (m/s$^2)$')
ax[2].set_xlabel('Radius (km)')
ax[2].set_ylim([0, 5])

# Make a subplot showing the calculated temperature profile
ax[3].plot(planet_mars.radii / 1.e3, planet_mars.temperature)
ax[3].set_ylabel('Temperature (K)')
ax[3].set_xlabel('Radius (km)')
ax[3].set_ylim([0., 5000])

for i in range(2):
    ax[i].set_xticklabels([])

for i in range(4):
    ax[i].set_xlim(0., max(planet_mars.radii) / 1.e3)
    # ax[i].set_ylim(0., maxy[i])

fig.set_tight_layout(True)
plt.savefig(f'{path_plot}/burnman.png')
plt.show()

dict = {
        'gravity': planet_mars.gravity,
        'pressure': planet_mars.pressure,
        'radius': planet_mars.radii,
        'temperature': planet_mars.temperature,
        'density': planet_mars.density,
        'mass planet': planet_mars.mass,
        'Normalized MoI': planet_mars.moment_of_inertia_factor
    }

path_data = f'{path}/data.pkl'
with open(path_data, 'wb') as f:
    pickle.dump(dict, f)


print(f'mass = {planet_mars.mass:.3e}')
print(f'moment of inertia factor= {planet_mars.moment_of_inertia_factor:.4f}')
