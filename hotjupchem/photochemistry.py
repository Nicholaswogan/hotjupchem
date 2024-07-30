import numpy as np
import numba as nb
import os
from scipy import constants as const
from tempfile import NamedTemporaryFile

from photochem import EvoAtmosphere, PhotoException
from photochem.utils._format import yaml, FormatSettings_main, MyDumper, Loader

from . import utils

DATA_DIR = os.path.dirname(os.path.realpath(__file__))+'/data/'

@nb.cfunc(nb.double(nb.double, nb.double, nb.double))
def custom_binary_diffusion_fcn(mu_i, mubar, T):
    # Equation 6 in Gladstone et al. (1996)
    b = 3.64e-5*T**(1.75-1.0)*7.3439e21*np.sqrt(2.01594/mu_i)
    return b

class EvoAtmosphereHJ(EvoAtmosphere):

    def __init__(self, mechanism_file, stellar_flux_file, planet_mass, planet_radius, 
                 nz=100, photon_scale_factor=1.0, P_ref=1.0e6, thermo_file=None):
        """Initializes the code

        Parameters
        ----------
        mechanism_file : str
            Path to the file describing the reaction mechanism
        stellar_flux_file : str
            Path to the file describing the stellar UV flux.
        planet_mass : float
            Planet mass in grams
        planet_radius : float
            Planet radius in cm
        nz : int, optional
            The number of layers in the photochemical model, by default 100
        P_ref : float, optional
            Pressure level corresponding to the planet_radius, by default 1e6 dynes/cm^2
        thermo_file: str, optional
            Optionally include a dedicated thermodynamic file.
        """        
        
        # First, initialize photochemical model with dummy inputs
        with open(DATA_DIR+'settings_template.yaml','r') as f:
            sol = yaml.load(f, Loader=Loader)
        sol['atmosphere-grid']['number-of-layers'] = int(nz)
        sol['planet']['planet-mass'] = float(planet_mass)
        sol['planet']['planet-radius'] = float(planet_radius)
        sol['planet']['photon-scale-factor'] = float(photon_scale_factor)
        sol = FormatSettings_main(sol)
        with NamedTemporaryFile('w') as f:
            yaml.dump(sol,f,Dumper=MyDumper)
            super().__init__(
                mechanism_file,
                f.name,
                stellar_flux_file,
                DATA_DIR+'atmosphere_init.txt'
            )

        # Save inputs that matter
        self.planet_radius = planet_radius
        self.planet_mass = planet_mass
        self.P_ref = P_ref

        # Parameters using during initialization
        # The factor of pressure the atmosphere extends
        # compared to predicted quench points of gases
        self.BOA_pressure_factor = 10.0
        # If True, then the guessed initial condition will used
        # quenching relations as an initial guess
        self.initial_cond_with_quenching = True
        # For computing chemical equilibrium at a metallicity.
        if thermo_file is None:
            thermo_file = mechanism_file
        self.m = utils.Metallicity(thermo_file)

        # Parameters for determining steady state
        self.TOA_pressure_avg = 1.0e-7*1e6 # mean TOA pressure (dynes/cm^2)
        self.max_dT_tol = 5 # The permitted difference between T in photochem and desired T
        self.max_dlog10edd_tol = 0.2 # The permitted difference between Kzz in photochem and desired Kzz
        self.freq_update_PTKzz = 1000 # step frequency to update PTKzz profile.
        self.max_total_step = 100_000 # Maximum total allowed steps before giving up
        self.min_step_conv = 300 # Min internal steps considered before convergence is allowed
        self.verbose = False # print information or not?
        self.freq_print = 100 # Frequency in which to print

        # Values in photochem to adjust
        self.var.upwind_molec_diff = True
        self.var.autodiff = True # Turn on autodiff
        self.var.atol = 1.0e-18
        self.var.conv_min_mix = 1e-10 # Min mix to consider during convergence check
        self.var.conv_longdy = 0.01 # threshold relative change that determines convergence
        self.var.custom_binary_diffusion_fcn = custom_binary_diffusion_fcn

        # Values that will be needed later. All of these set
        # in `initialize_to_climate_equilibrium_PT`
        self.P_clima_grid = None # The climate grid
        self.metallicity = None
        self.CtoO = None
        # Below for interpolation
        self.log10P_interp = None
        self.T_interp = None
        self.log10edd_interp = None
        self.P_desired = None
        self.T_desired = None
        self.Kzz_desired = None
        # Index of climate grid that is bottom of photochemical grid
        self.ind_b = None
        # information needed during robust stepping
        self.total_step_counter = None
        self.nerrors = None
        self.robust_stepper_initialized = None

    def initialize_to_climate_equilibrium_PT(self, P_in, T_in, Kzz_in, metallicity, CtoO, rainout_condensed_atoms=True):
        """Initialized the photochemical model to a climate model result that assumes chemical equilibrium
        at some metallicity and C/O ratio.

        Parameters
        ----------
        P_in : ndarray[dim=1,double]
            The pressures in the climate grid (dynes/cm^2). P_in[0] is pressure at
            the deepest layer of the atmosphere
        T_in : ndarray[dim1,double]
            The temperatures in the climate grid corresponding to P_in (K)
        Kzz_in : ndarray[dim1,double]
            The eddy diffusion at each pressure P_in (cm^2/s)
        metallicity : float
            Metallicity relative to solar.
        CtoO : float
            C/O ratio relative to solar. So CtoO = 1 is solar C/O ratio.
            CtoO = 2 is twice the solar C/O ratio.
        """

        if P_in.shape[0] != T_in.shape[0]:
            raise Exception('Input P and T must have same shape')
        if P_in.shape[0] != Kzz_in.shape[0]:
            raise Exception('Input P and Kzz must have same shape')

        # Save inputs
        self.P_clima_grid = P_in
        self.metallicity = metallicity
        self.CtoO = CtoO

        # Compute chemical equilibrium along the whole P-T profile
        mix, mubar = self.m.composition(T_in, P_in, CtoO, metallicity, rainout_condensed_atoms)

        if self.TOA_pressure_avg*3 > P_in[-1]:
            raise Exception('The photochemical grid needs to extend above the climate grid')

        # Altitude of P-T grid
        P1, T1, mubar1, z1 = utils.compute_altitude_of_PT(P_in, self.P_ref, T_in, mubar, self.planet_radius, self.planet_mass, self.TOA_pressure_avg)
        # If needed, extrapolate Kzz and mixing ratios
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

        # Next, we compute the quench levels
        quench_levels = utils.determine_quench_levels(T1, P1, Kzz1, mubar1, grav1)
        ind = np.min(quench_levels) # the deepest quench level

        # If desired, this bit applies quenched initial conditions, and recomputes
        # the altitude profile for this new mubar.
        if self.initial_cond_with_quenching:

            # Apply quenching to mixing ratios
            mix1['CH4'][quench_levels[0]:] = mix1['CH4'][quench_levels[0]]
            mix1['CO'][quench_levels[0]:] = mix1['CO'][quench_levels[0]]
            mix1['CO2'][quench_levels[1]:] = mix1['CO2'][quench_levels[1]]
            mix1['NH3'][quench_levels[2]:] = mix1['NH3'][quench_levels[2]]
            mix1['HCN'][quench_levels[3]:] = mix1['HCN'][quench_levels[3]]

            # Quenching out H2 at the CH4 level seems to work well
            mix1['H2'][quench_levels[0]:] = mix1['H2'][quench_levels[0]]

            # Normalize mixing ratios
            mix_tot = np.zeros(mix1['CH4'].shape[0])
            for key in mix1:
                mix_tot += mix1[key]
            for key in mix1:
                mix1[key] = mix1[key]/mix_tot

            # Compute mubar again
            mubar1[:] = 0.0
            for i,sp in enumerate(self.dat.species_names[:-2]):
                if sp in mix1:
                    for j in range(P1.shape[0]):
                        mubar1[j] += mix1[sp][j]*self.dat.species_mass[i]

            # Update z1 to get a new altitude profile
            P1, T1, mubar1, z1 = utils.compute_altitude_of_PT(P1, self.P_ref, T1, mubar1, self.planet_radius, self.planet_mass, self.TOA_pressure_avg)

        # Save P-T-Kzz for later interpolation and corrections
        self.log10P_interp = np.log10(P1.copy()[::-1])
        self.T_interp = T1.copy()[::-1]
        self.log10edd_interp = np.log10(Kzz1.copy()[::-1])
        self.P_desired = P1.copy()
        self.T_desired = T1.copy()
        self.Kzz_desired = Kzz1.copy()

        # Bottom of photochemical model will be at a pressure a factor
        # larger than the predicted quench pressure.
        if P1[ind]*self.BOA_pressure_factor > P1[0]:
            raise Exception('BOA in photochemical model wants to be deeper than BOA of climate model.')
        self.ind_b = np.argmin(np.abs(P1 - P1[ind]*self.BOA_pressure_factor))
        
        self._initialize_atmosphere(P1, T1, Kzz1, z1, mix1)

    def reinitialize_to_new_climate_PT(self, P_in, T_in, Kzz_in, mix):
        """Reinitializes the photochemical model to the input P, T, Kzz, and mixing ratios
        from the climate model.

        Parameters
        ----------
        P_in : ndarray[ndim=1,double]
            Pressure grid in climate model (dynes/cm^2).
        T_in : ndarray[ndim=1,double]
            Temperatures corresponding to P_in (K)
        Kzz_in : ndarray[ndim,double]
            Eddy diffusion coefficients at each pressure level (cm^2/s)
        mix : dict
            Mixing ratios of all species in the atmosphere

        """        

        if self.P_clima_grid is None:
            raise Exception('This routine can only be called after `initialize_to_climate_equilibrium_PT`')
        if not np.all(np.isclose(self.P_clima_grid,P_in)):
            raise Exception('Input pressure grid does not match saved pressure grid')
        if P_in.shape[0] != T_in.shape[0]:
            raise Exception('Input P and T must have same shape')
        if P_in.shape[0] != Kzz_in.shape[0]:
            raise Exception('Input P and Kzz must have same shape')
        for key in mix:
            if P_in.shape[0] != mix[key].shape[0]:
                raise Exception('Input P and mix must have same shape')
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        if set(list(mix.keys())) != set(species_names):
            raise Exception('Some species are missing from input mix') 
        
        # Compute mubar
        mubar = np.zeros(T_in.shape[0])
        species_mass = self.dat.species_mass
        particle_names = self.dat.species_names[:self.dat.np]
        for sp in mix:
            if sp not in particle_names:
                ind = species_names.index(sp)
                mubar = mubar + mix[sp]*species_mass[ind]

        # Compute altitude of P-T grid
        P1, T1, mubar1, z1 = utils.compute_altitude_of_PT(P_in, self.P_ref, T_in, mubar, self.planet_radius, self.planet_mass, self.TOA_pressure_avg)
        # If needed, extrapolte Kzz and mixing ratios
        if P1.shape[0] != Kzz_in.shape[0]:
            Kzz1 = np.append(Kzz_in,Kzz_in[-1])
            mix1 = {}
            for sp in mix:
                mix1[sp] = np.append(mix[sp],mix[sp][-1])
        else:
            Kzz1 = Kzz_in.copy()
            mix1 = mix

        # Save P-T-Kzz for later interpolation and corrections
        self.log10P_interp = np.log10(P1.copy()[::-1])
        self.T_interp = T1.copy()[::-1]
        self.log10edd_interp = np.log10(Kzz1.copy()[::-1])
        self.P_desired = P1.copy()
        self.T_desired = T1.copy()
        self.Kzz_desired = Kzz1.copy()

        self._initialize_atmosphere(P1, T1, Kzz1, z1, mix1)

    def _initialize_atmosphere(self, P1, T1, Kzz1, z1, mix1):
        "Little helper function preventing code duplication."

        # Compute TOA index
        ind_t = np.argmin(np.abs(P1 - self.TOA_pressure_avg))

        # Shift z profile so that zero is at photochem BOA
        z1_p = z1 - z1[self.ind_b]

        # Calculate the photochemical grid
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
        planet_radius_new = self.planet_radius + z1[self.ind_b]

        # Update photochemical model grid
        self.dat.planet_radius = planet_radius_new
        self.update_vertical_grid(TOA_alt=z_top) # this will update gravity for new planet radius
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
            if i >= self.dat.np:
                self.set_lower_bc(sp, bc_type='Moses') # gas
            else:
                self.set_lower_bc(sp, bc_type='vdep', vdep=0.0) # particle
        particle_names = self.dat.species_names[:self.dat.np]
        for sp in mix_p:
            if sp not in particle_names:
                Pi = P_p[0]*mix_p[sp][0]
                self.set_lower_bc(sp, bc_type='press', press=Pi)

        self.prep_atmosphere(self.wrk.usol)

    def return_atmosphere_climate_grid(self):
        """Returns a dictionary with temperature, Kzz and mixing ratios
        on the climate model grid.

        Returns
        -------
        dict
            Contains temperature, Kzz, and mixing ratios.
        """ 
        if self.P_clima_grid is None:
            raise Exception('This routine can only be called after `initialize_to_climate_equilibrium_PT`')

        # return full atmosphere
        out = self.return_atmosphere()

        # Interpolate full atmosphere to clima grid
        sol = {}
        sol['pressure'] = self.P_clima_grid.copy()
        log10Pclima = np.log10(self.P_clima_grid[::-1]).copy()
        log10P = np.log10(out['pressure'][::-1]).copy()

        T = np.interp(log10Pclima, log10P, out['temperature'][::-1].copy())
        sol['temperature'] = T[::-1].copy()

        Kzz = np.interp(log10Pclima, log10P, np.log10(out['Kzz'][::-1].copy()))
        sol['Kzz'] = 10.0**Kzz[::-1].copy()

        for key in out:
            if key not in ['pressure','temperature','Kzz']:
                tmp = np.log10(np.clip(out[key][::-1].copy(),a_min=1e-100,a_max=np.inf))
                mix = np.interp(log10Pclima, log10P, tmp)
                sol[key] = 10.0**mix[::-1].copy()

        return sol

    def return_atmosphere(self, include_deep_atmosphere = True, equilibrium = False, rainout_condensed_atoms = True):
        """Returns a dictionary with temperature, Kzz and mixing ratios
        on the photochemical grid.

        Parameters
        ----------
        include_deep_atmosphere : bool, optional
            If True, then results will include portions of the deep
            atomsphere that are not part of the photochemical grid, by default True

        Returns
        -------
        dict
            Contains temperature, Kzz, and mixing ratios.
        """        

        if self.P_clima_grid is None:
            raise Exception('This routine can only be called after `initialize_to_climate_equilibrium_PT`')

        out = {}
        out['pressure'] = self.wrk.pressure_hydro
        out['temperature'] = self.var.temperature
        out['Kzz'] = self.var.edd
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        if equilibrium:
            mix, mubar = self.m.composition(out['temperature'], out['pressure'], self.CtoO, self.metallicity, rainout_condensed_atoms)
            for key in mix:
                out[key] = mix[key]
            for key in species_names[:self.dat.np]:
                out[key] = np.zeros(mix['H2'].shape[0])
        else:
            for i,sp in enumerate(species_names):
                mix = self.wrk.usol[i,:]/self.wrk.density
                out[sp] = mix

        if not include_deep_atmosphere:
            return out

        # Prepend the deeper atmosphere, which we will assume is at Equilibrium
        inds = np.where(self.P_desired > self.wrk.pressure_hydro[0])
        out1 = {}
        out1['pressure'] = self.P_desired[inds]
        out1['temperature'] = self.T_desired[inds]
        out1['Kzz'] = self.Kzz_desired[inds]
        mix, mubar = self.m.composition(out1['temperature'], out1['pressure'], self.CtoO, self.metallicity, rainout_condensed_atoms)
        
        out['pressure'] = np.append(out1['pressure'],out['pressure'])
        out['temperature'] = np.append(out1['temperature'],out['temperature'])
        out['Kzz'] = np.append(out1['Kzz'],out['Kzz'])
        for i,sp in enumerate(species_names):
            if sp in mix:
                out[sp] = np.append(mix[sp],out[sp])
            else:
                out[sp] = np.append(np.zeros(mix['H2'].shape[0]),out[sp])

        return out
    
    def initialize_robust_stepper(self, usol):
        """Initialized a robust integrator.

        Parameters
        ----------
        usol : ndarray[double,dim=2]
            Input number densities
        """        
        if self.P_clima_grid is None:
            raise Exception('This routine can only be called after `initialize_to_climate_equilibrium_PT`')
        
        self.total_step_counter = 0
        self.nerrors = 0
        self.initialize_stepper(usol)
        self.robust_stepper_initialized = True

    def robust_step(self):
        """Takes a single robust integrator step

        Returns
        -------
        tuple
            The tuple contains two bools `give_up, reached_steady_state`. If give_up is True
            then the algorithm things it is time to give up on reaching a steady state. If
            reached_steady_state then the algorithm has reached a steady state within
            tolerance.
        """        
        if self.P_clima_grid is None:
            raise Exception('This routine can only be called after `initialize_to_climate_equilibrium_PT`')

        if not self.robust_stepper_initialized:
            raise Exception('This routine can only be called after `initialize_robust_stepper`')

        give_up = False
        reached_steady_state = False

        for i in range(1):
            try:
                self.step()
                self.total_step_counter += 1
            except PhotoException as e:
                # If there is an error, lets reinitialize, but get rid of any
                # negative numbers
                usol = np.clip(self.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf)
                self.initialize_stepper(usol)
                self.nerrors += 1

                if self.nerrors > 10:
                    give_up = True
                    break

            # convergence checking
            converged = self.check_for_convergence()

            # Compute the max difference between the P-T profile in photochemical model
            # and the desired P-T profile
            T_p = np.interp(np.log10(self.wrk.pressure_hydro.copy()[::-1]), self.log10P_interp, self.T_interp)
            T_p = T_p.copy()[::-1]
            max_dT = np.max(np.abs(T_p - self.var.temperature))

            # Compute the max difference between the P-edd profile in photochemical model
            # and the desired P-edd profile
            log10edd_p = np.interp(np.log10(self.wrk.pressure_hydro.copy()[::-1]), self.log10P_interp, self.log10edd_interp)
            log10edd_p = log10edd_p.copy()[::-1]
            max_dlog10edd = np.max(np.abs(log10edd_p - np.log10(self.var.edd)))

            # TOA pressure
            TOA_pressure = self.wrk.pressure_hydro[-1]

            condition1 = converged and self.wrk.nsteps > self.min_step_conv or self.wrk.tn > self.var.equilibrium_time
            condition2 = max_dT < self.max_dT_tol and max_dlog10edd < self.max_dlog10edd_tol and self.TOA_pressure_avg/3 < TOA_pressure < self.TOA_pressure_avg*3

            if condition1 and condition2:
                if self.verbose:
                    print('nsteps = %i  longdy = %.1e  max_dT = %.1e  max_dlog10edd = %.1e  TOA_pressure = %.1e'% \
                        (self.total_step_counter, self.wrk.longdy, max_dT, max_dlog10edd, TOA_pressure/1e6))
                # success!
                reached_steady_state = True
                break

            if not (self.wrk.nsteps % self.freq_update_PTKzz) or (condition1 and not condition2):
                # After ~1000 steps, lets update P,T, edd and vertical grid
                self.set_press_temp_edd(self.P_desired,self.T_desired,self.Kzz_desired,hydro_pressure=True)
                self.update_vertical_grid(TOA_pressure=self.TOA_pressure_avg)
                self.initialize_stepper(self.wrk.usol)

            if self.total_step_counter > self.max_total_step:
                give_up = True
                break

            if not (self.wrk.nsteps % self.freq_print) and self.verbose:
                print('nsteps = %i  longdy = %.1e  max_dT = %.1e  max_dlog10edd = %.1e  TOA_pressure = %.1e'% \
                    (self.total_step_counter, self.wrk.longdy, max_dT, max_dlog10edd, TOA_pressure/1e6))
                
        return give_up, reached_steady_state
    
    def find_steady_state(self):
        """Attempts to find a photochemical steady state.

        Returns
        -------
        bool
            If True, then the routine was successful.
        """    

        self.initialize_robust_stepper(self.wrk.usol)
        success = True
        while True:
            give_up, reached_steady_state = self.robust_step()
            if reached_steady_state:
                break
            if give_up:
                success = False
                break
        return success
    
    def model_state_to_dict(self):
        """Returns a dictionary containing all information needed to reinitialize the atmospheric
        state. This dictionary can be used as an input to "initialize_from_dict".
        """

        if self.P_clima_grid is None:
            raise Exception('This routine can only be called after `initialize_to_climate_equilibrium_PT`')

        out = {}
        out['P_clima_grid'] = self.P_clima_grid
        out['metallicity'] = self.metallicity
        out['CtoO'] = self.CtoO
        out['log10P_interp'] = self.log10P_interp
        out['T_interp'] = self.T_interp
        out['log10edd_interp'] = self.log10edd_interp
        out['P_desired'] = self.P_desired
        out['T_desired'] = self.T_desired
        out['Kzz_desired'] = self.Kzz_desired
        out['ind_b'] = self.ind_b
        out['planet_radius_new'] = self.dat.planet_radius
        out['top_atmos'] = self.var.top_atmos
        out['temperature'] = self.var.temperature
        out['edd'] = self.var.edd
        out['usol'] = self.wrk.usol
        out['P_i_surf'] = (self.wrk.usol[self.dat.np:,0]/self.wrk.density[0])*self.wrk.pressure[0]

        return out

    def initialize_from_dict(self, out):
        """Initializes the model from a dictionary created by the "model_state_to_dict" routine.
        """

        self.P_clima_grid = out['P_clima_grid']
        self.metallicity = out['metallicity']
        self.CtoO = out['CtoO']
        self.log10P_interp = out['log10P_interp']
        self.T_interp = out['T_interp']
        self.log10edd_interp = out['log10edd_interp']
        self.P_desired = out['P_desired']
        self.T_desired = out['T_desired']
        self.Kzz_desired = out['Kzz_desired']
        self.ind_b = out['ind_b']
        self.dat.planet_radius = out['planet_radius_new']
        self.update_vertical_grid(TOA_alt=out['top_atmos'])
        self.set_temperature(out['temperature'])
        self.var.edd = out['edd']
        self.wrk.usol = out['usol']

        # Now set boundary conditions
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        for i,sp in enumerate(species_names):
            if i >= self.dat.np:
                self.set_lower_bc(sp, bc_type='Moses') # gas
            else:
                self.set_lower_bc(sp, bc_type='vdep', vdep=0.0) # particle
        species_names = self.dat.species_names[self.dat.np:(-2-self.dat.nsl)]
        for i,sp in enumerate(species_names):
            self.set_lower_bc(sp, bc_type='press', press=out['P_i_surf'][i])

        self.prep_atmosphere(self.wrk.usol)
