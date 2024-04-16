import numpy as np
import numba as nb
from scipy import constants as const
from scipy import integrate
import cantera as ct

def composition_from_metallicity(sun_mol, M_H_metallicity):
    """Returns composition given metallicity

    Parameters
    ----------
    sun_mol : dict
        Elemental composition of the sun (mol/mol).
    M_H_metallicity : float
        log10 metallicity relative to solar

    Returns
    -------
    dict
        Contains atomic composition.
    """

    # Check that H and He are in sun's compositions
    if 'H' not in sun_mol:
        raise Exception('"H" must be part of solar composition.')
    if 'He' not in sun_mol:
        raise Exception('"He" must be part of solar composition.')

    # Separate metals from non-metals
    metals = []
    for key in sun_mol:
        if key != 'H' and key != 'He':
            metals.append(key)

    # Add up all metals
    mol_metals = 0.0
    for sp in metals:
        mol_metals += sun_mol[sp]

    # Compute the mol of H, metal and He of body
    mol_H_body = (10.0**M_H_metallicity * (mol_metals/sun_mol['H']) + 1.0 + sun_mol['He']/sun_mol['H'])**(-1.0)
    mol_metal_body = 10.0**M_H_metallicity * (mol_metals/sun_mol['H'])*mol_H_body
    mol_He_body = (sun_mol['He']/sun_mol['H'])*mol_H_body
    
    # Check everything worked out
    assert np.isclose((mol_metal_body/mol_H_body)/(mol_metals/sun_mol['H']), 10.0**M_H_metallicity)
    
    # Get metal composition
    metal_fractions = {}
    for sp in metals:
        metal_fractions[sp] = sun_mol[sp]/mol_metals

    # compute composition of the body
    mol_body = {}
    mol_body['H'] = mol_H_body
    mol_body['He'] = mol_He_body
    for sp in metals:
        mol_body[sp] = mol_metal_body*metal_fractions[sp]

    return mol_body

def composition_from_metallicity_for_atoms(atoms, sun_mol, M_H_metallicity):
    """Computes composition given metallicity and a list of atoms.

    Parameters
    ----------
    atoms : list
        List of atoms
    sun_mol : dict
        Elemental composition of the sun (mol/mol).
    M_H_metallicity : float
        log10 metallicity relative to solar

    Returns
    -------
    dict
        Contains atomic composition.
    """    
    mol_body = composition_from_metallicity(sun_mol, M_H_metallicity)

    mol_tot = 0.0
    for atom in atoms:
        if atom in mol_body:
            mol_tot += mol_body[atom]

    mol_out = {}
    for atom in atoms:
        if atom in mol_body: 
            mol_out[atom] = mol_body[atom]/mol_tot
        else:
            mol_out[atom] = 0.0
    
    return mol_out

class Metallicity():

    def __init__(self, ct_file):
        """A simple Metallicity calculator.

        Parameters
        ----------
        ct_file : str
            Path to a Cantera input file.
        """        
        # composition of the Sun (mol/mol)
        # From Table 8 in Lodders et al. (2009), 
        # "Abundances of the elements in the solar system"
        self.sun_mol = {
            'H': 0.921514888949834,
            'He': 0.07749066995740882,
            'O': 0.0004946606569284939,
            'C': 0.0002300986287941852,
            'N': 6.278165064228584e-05,
            'Si': 3.131024001891741e-05,
            'Mg': 3.101174053726646e-05,
            'Ne': 0.00010582900979606936,
            'Fe': 2.6994013922759826e-05,
            'S': 1.1755152117252983e-05
        }

        self.gas = ct.Solution(ct_file)
        if 'H' not in self.gas.element_names:
            raise Exception('"H" must be an element in the Cantera file')
        if 'He' not in self.gas.element_names:
            raise Exception('"He" must be an element in the Cantera file')

    def composition(self, T, P, CtoO, metal):
        """Given a T-P profile, C/O ratio and metallicity, the code
        computes chemical equilibrium composition.

        Parameters
        ----------
        T : ndarray[dim=1,float64]
            Temperature in K
        P : ndarray[dim=1,float64]
            Pressure in dynes/cm^2
        CtoO : float
            The C / O ratio relative to solar. CtoO = 1 would be the same
            composition as solar.
        metal : float
            Metallicity relative to solar.

        Returns
        -------
        dict
            Composition at chemical equilibrium.
        """

        # Check T and P
        if isinstance(T, float) or isinstance(T, int):
            T = np.array([T],np.float64)
        if isinstance(P, float) or isinstance(P, int):
            P = np.array([P],np.float64)
        if not isinstance(P, np.ndarray):
            raise ValueError('"P" must by an np.ndarray')
        if not isinstance(T, np.ndarray):
            raise ValueError('"P" must by an np.ndarray')
        if T.ndim != 1:
            raise ValueError('"T" must have one dimension')
        if P.ndim != 1:
            raise ValueError('"P" must have one dimension')
        if T.shape[0] != P.shape[0]:
            raise ValueError('"P" and "T" must be the same length')
        # Check CtoO and metal
        if CtoO <= 0:
            raise ValueError('"CtoO" must be greater than 0')
        if metal <= 0:
            raise ValueError('"metal" must be greater than 0')

        # Get composition
        comp = composition_from_metallicity_for_atoms(self.gas.element_names, self.sun_mol, np.log10(metal))

        # Adjust C and O to get desired C/O ratio. CtoO is relative to solar
        x = CtoO*(comp['C']/comp['O'])
        a = (x*comp['O'] - comp['C'])/(1+x)
        comp['C'] = comp['C'] + a
        comp['O'] = comp['O'] - a

        # For output
        out = {}
        for sp in self.gas.species_names:
            out[sp] = np.empty(P.shape[0])
        mubar = np.empty(P.shape[0])

        # Compute chemical equilibrium at all altitudes
        for i in range(P.shape[0]):
            self.gas.TPX = T[i],(P[i]/1.0e6)*1e5,comp
            self.gas.equilibrate('TP')
            for j,sp in enumerate(self.gas.species_names):
                out[sp][i] = self.gas.X[j]
            mubar[i] = self.gas.mean_molecular_weight

        return out, mubar
    
@nb.njit()
def CH4_CO_quench_timescale(T, P):
    "T in K, P in dynes/cm^2, tq in s. Equation 11."
    P_bars = P/1.0e6
    tq = 3.0e-6*P_bars**-1*np.exp(42_000.0/T)
    return tq

@nb.njit()
def NH3_quench_timescale(T, P):
    "T in K, P in dynes/cm^2, tq in s. Equation 32."
    P_bars = P/1.0e6
    tq = 1.0e-7*P_bars**-1*np.exp(52_000.0/T)
    return tq

@nb.njit()
def HCN_quench_timescale(T, P):
    "T in K, P in dynes/cm^2, tq in s. From PICASO."
    P_bars = P/1.0e6
    tq = (1.5e-4/(P_bars*(3.0**0.7)))*np.exp(36_000.0/T)
    return tq

@nb.njit()
def CO2_quench_timescale(T, P):
    "T in K, P in dynes/cm^2, tq in s. Equation 44."
    P_bars = P/1.0e6
    tq = 1.0e-10*P_bars**-0.5*np.exp(38_000.0/T)
    return tq

@nb.njit()
def scale_height(T, mubar, grav):
    "All inputs are CGS."
    k_boltz = const.k*1e7
    H = (const.Avogadro*k_boltz*T)/(mubar*grav)
    return H

@nb.njit()
def determine_quench_levels(T, P, Kzz, mubar, grav):

    # Mixing timescale
    tau_mix = scale_height(T, mubar, grav)**2/Kzz

    # Quenching timescales
    tau_CH4 = CH4_CO_quench_timescale(T, P)
    tau_CO2 = CO2_quench_timescale(T, P)
    tau_NH3 = NH3_quench_timescale(T, P)
    tau_HCN = HCN_quench_timescale(T, P)

    # Quench level is when the chemistry timescale
    # exceeds the mixing timescale.
    quench_levels = np.zeros(4, dtype=np.int32)
    
    for i in range(P.shape[0]):
        quench_levels[0] = i
        if tau_CH4[i] > tau_mix[i]:
            break

    for i in range(P.shape[0]):
        quench_levels[1] = i
        if tau_CO2[i] > tau_mix[i]:
            break

    for i in range(P.shape[0]):
        quench_levels[2] = i
        if tau_NH3[i] > tau_mix[i]:
            break

    for i in range(P.shape[0]):
        quench_levels[3] = i
        if tau_HCN[i] > tau_mix[i]:
            break

    return quench_levels

@nb.njit()
def deepest_quench_level(T, P, Kzz, mubar, grav):
    quench_levels = determine_quench_levels(T, P, Kzz, mubar, grav)
    return np.min(quench_levels)
    

class TempPressMubar:

    def __init__(self, P, T, mubar):
        self.log10P = np.log10(P)[::-1].copy()
        self.T = T[::-1].copy()
        self.mubar = mubar[::-1].copy()

    def temperature_mubar(self, P):
        T = np.interp(np.log10(P), self.log10P, self.T)
        mubar = np.interp(np.log10(P), self.log10P, self.mubar)
        return T, mubar
    
def gravity(radius, mass, z):
    G_grav = const.G
    grav = G_grav * (mass/1.0e3) / ((radius + z)/1.0e2)**2.0
    grav = grav*1.0e2 # convert to cgs
    return grav

def hydrostatic_equation(P, u, planet_radius, planet_mass, ptm):
    z = u[0]
    grav = gravity(planet_radius, planet_mass, z)
    T, mubar = ptm.temperature_mubar(P)
    k_boltz = const.Boltzmann*1e7
    dz_dP = -(k_boltz*T*const.Avogadro)/(mubar*grav*P)
    return np.array([dz_dP])

def compute_altitude_of_PT(P, P_ref, T, mubar, planet_radius, planet_mass, P_top):
    ptm = TempPressMubar(P, T, mubar)
    args = (planet_radius, planet_mass, ptm)
    if P_top < P[-1]:
        P_top_ = P_top
        P_ = np.append(P,P_top_)
        T_ = np.append(T,T[-1])
        mubar_ = np.append(mubar,mubar[-1])
    else:
        P_top_ = P[-1]
        P_ = P.copy()
        T_ = T.copy()
        mubar_ = mubar.copy()

    if P_ref > P_[0] or P_ref < P_[-1]:
        raise Exception('Reference pressure must be within P grid.')
    ind_ref = np.argmin(np.abs(P_ - P_ref))

    if ind_ref == 0 or ind_ref == P_.shape[0]:
        raise Exception('Reference pressure mest be within P grid.')

    out2 = integrate.solve_ivp(hydrostatic_equation, [P_[ind_ref], P_[-1]], np.array([0.0]), t_eval=P_[ind_ref:], args=args)
    out1 = integrate.solve_ivp(hydrostatic_equation, [P_[ind_ref], P_[0]], np.array([0.0]), t_eval=P_[:ind_ref][::-1], args=args)

    z_ = np.append(out1.y[0][::-1],out2.y[0])
    return P_, T_, mubar_, z_

