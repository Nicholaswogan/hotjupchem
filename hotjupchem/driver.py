import numpy as np
import os
from scipy import constants as const

from photochem import EvoAtmosphere, PhotoException
from photochem.utils._format import yaml, FormatSettings_main, MyDumper, Loader

from . import utils

DATA_DIR = os.path.dirname(os.path.realpath(__file__))+'/data/'

class EvoAtmosphereHJ(EvoAtmosphere):

    stellar_flux_file : str
    planet_radius : float
    planet_mass : float
    TOA_pressure_avg : float
    TOA_pressure_min : float
    TOA_pressure_max : float

    def __init__(self, stellar_flux_file, planet_mass, planet_radius, nz=50):

        self.stellar_flux_file = stellar_flux_file
        self.planet_radius = planet_radius
        self.planet_mass = planet_mass
        self.TOA_pressure_avg = 1.0e-7*1e6
        self.TOA_pressure_min = 1.0e-8*1e6
        self.TOA_pressure_max = 5.0e-7*1e6

        # Using settings template
        with open(DATA_DIR+'settings_template.yaml','r') as f:
            sol = yaml.load(f, Loader=Loader)
        sol['atmosphere-grid']['number-of-layers'] = int(nz)
        sol['planet']['planet-mass'] = float(planet_mass)
        sol['planet']['planet-radius'] = float(planet_radius)
        sol = FormatSettings_main(sol)
        with open('tmpfile1234567890.yaml', 'w') as f:
            yaml.dump(sol,f,Dumper=MyDumper)

        EvoAtmosphere.__init__(
            self,
            DATA_DIR+'zahnle_earth_HNOCS.yaml',
            'tmpfile1234567890.yaml',
            stellar_flux_file,
            DATA_DIR+'atmosphere_init.txt'
        )
        os.remove('tmpfile1234567890.yaml')

        # Adjust default values
        self.var.autodiff = True # Turn on autodiff
        self.var.atol = 1e-14 # Seems to be a good
        self.var.rtol = 1e-3
        self.var.conv_min_mix = 1e-10
        self.var.conv_longdy = 0.05 

    def initialize_to_PTX(self, P_in, P_ref, T_in, Kzz_in, metallicity, CtoO):

        # Compute chemical equilibrium along the whole P-T profile
        m = utils.Metallicity(DATA_DIR+'zahnle_earth_HNOCS_ct.yaml')
        mix, mubar = m.composition(T_in, P_in, CtoO, metallicity)

        # Altitude of P-T grid
        P1, T1, mubar1, z1 = utils.compute_altitude_of_PT(P_in, P_ref, T_in, mubar, self.planet_radius, self.planet_mass, self.TOA_pressure_avg)
        # If needed, extrapolte Kzz and mixing ratios
        if P1.shape[0] != Kzz_in.shape[0]:
            Kzz1 = np.append(Kzz_in,Kzz_in[-1])
            mix1 = {}
            for sp in mix:
                mix1[sp] = np.append(mix[sp],mix[sp][-1])
        else:
            Kzz1 = Kzz_in.copy()
            mix1 = mix

        # The gravity
        grav1 = utils.gravity(self.planet_radius, self.planet_mass, z1)

        # Next, we compute the deepest quench level
        ind = utils.deepest_quench_level(T1, P1, Kzz1, mubar1, grav1)

        # Bottom of photochemical model will be at a pressure a factor of 10
        # larger than the predicted quench pressure.
        ind_b = np.argmin(np.abs(P1 - P1[ind]*10.0))
        
        # Compute TOA index
        ind_t = np.argmin(np.abs(P1 - self.TOA_pressure_avg))

        # Shift z profile so that zero is at photochem BOA
        z1_p = z1 - z1[ind_b]

        # Now we compute where the photochem grid points will be
        z_top = z1_p[ind_t]
        z_bottom = 0.0
        dz = (z_top - z_bottom)/self.var.nz
        z_p = np.empty(self.var.nz)
        z_p[0] = dz/2.0
        for i in range(1,self.var.nz):
            z_p[i] = z_p[i-1] + dz

        # Now, we interpolate all values to the photochemical grid
        P_p = 10.0**np.interp(z_p, z1_p, np.log10(P1))
        T_p = np.interp(z_p, z1_p, T1)
        Kzz_p = 10.0**np.interp(z_p, z1_p, np.log10(Kzz1))
        mix_p = {}
        for sp in mix1:
            mix_p[sp] = 10.0**np.interp(z_p, z1_p, np.log10(mix1[sp]))
        k_boltz = const.k*1e7
        den_p = P_p/(k_boltz*T_p)

        # Compute new planet radius
        planet_radius_new = self.planet_radius + z1[ind_b]

        # Update photochemical model grid
        self.update_planet_mass_radius(self.planet_mass, planet_radius_new)
        self.update_vertical_grid(TOA_alt=z_top)
        self.set_temperature(T_p)
        self.var.edd = Kzz_p
        usol = np.ones(self.wrk.usol.shape)*1e-40
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        for sp in mix_p:
            if sp in species_names:
                ind = species_names.index(sp)
                usol[ind,:] = mix_p[sp]*den_p
        self.wrk.usol = usol

        # Now set boundary conditions
        for i,sp in enumerate(species_names):
            if sp not in ['hv','M']:
                self.set_lower_bc(sp, bc_type='Moses')
        for sp in mix_p:
            Pi = P_p[0]*mix_p[sp][0]
            self.set_lower_bc(sp, bc_type='press', press=Pi)

        self.prep_atmosphere(self.wrk.usol)


        

    


