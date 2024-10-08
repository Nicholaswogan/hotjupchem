import numpy as np
import numba as nb
from numba import types
from scipy import constants as const
from scipy import integrate
from photochem import equilibrate
from picaso import fluxes
from matplotlib import pyplot as plt

class Metallicity():

    def __init__(self, filename):
        """A simple Metallicity calculator.

        Parameters
        ----------
        filename : str
            Path to a thermodynamic file
        """
        self.gas = equilibrate.ChemEquiAnalysis(filename)

    def composition(self, T, P, CtoO, metal, rainout_condensed_atoms = True):
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
        rainout_condensed_atoms : bool, optional
            If True, then the code will rainout atoms that condense.

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

        # For output
        out = {}
        for sp in self.gas.gas_names:
            out[sp] = np.empty(P.shape[0])
        mubar = np.empty(P.shape[0])
        
        molfracs_atoms = self.gas.molfracs_atoms_sun
        for i,sp in enumerate(self.gas.atoms_names):
            if sp != 'H' and sp != 'He':
                molfracs_atoms[i] = self.gas.molfracs_atoms_sun[i]*metal
        molfracs_atoms = molfracs_atoms/np.sum(molfracs_atoms)

        # Adjust C and O to get desired C/O ratio. CtoO is relative to solar
        indC = self.gas.atoms_names.index('C')
        indO = self.gas.atoms_names.index('O')
        x = CtoO*(molfracs_atoms[indC]/molfracs_atoms[indO])
        a = (x*molfracs_atoms[indO] - molfracs_atoms[indC])/(1+x)
        molfracs_atoms[indC] = molfracs_atoms[indC] + a
        molfracs_atoms[indO] = molfracs_atoms[indO] - a

        # Compute chemical equilibrium at all altitudes
        for i in range(P.shape[0]):
            self.gas.solve(P[i], T[i], molfracs_atoms=molfracs_atoms)
            for j,sp in enumerate(self.gas.gas_names):
                out[sp][i] = self.gas.molfracs_species_gas[j]
            mubar[i] = self.gas.mubar
            if rainout_condensed_atoms:
                molfracs_atoms = self.gas.molfracs_atoms_gas

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
    
@nb.experimental.jitclass()
class TempPressMubar:

    log10P : types.double[:]
    T : types.double[:]
    mubar : types.double[:]

    def __init__(self, P, T, mubar):
        self.log10P = np.log10(P)[::-1].copy()
        self.T = T[::-1].copy()
        self.mubar = mubar[::-1].copy()

    def temperature_mubar(self, P):
        T = np.interp(np.log10(P), self.log10P, self.T)
        mubar = np.interp(np.log10(P), self.log10P, self.mubar)
        return T, mubar

@nb.njit()
def gravity(radius, mass, z):
    G_grav = const.G
    grav = G_grav * (mass/1.0e3) / ((radius + z)/1.0e2)**2.0
    grav = grav*1.0e2 # convert to cgs
    return grav

@nb.njit()
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
        # If P_top is lower P than P grid, then we extend it
        P_top_ = P_top
        P_ = np.append(P,P_top_)
        T_ = np.append(T,T[-1])
        mubar_ = np.append(mubar,mubar[-1])
    else:
        P_top_ = P[-1]
        P_ = P.copy()
        T_ = T.copy()
        mubar_ = mubar.copy()

    # Make sure P_ref is in the P grid
    if P_ref > P_[0] or P_ref < P_[-1]:
        raise Exception('Reference pressure must be within P grid.')
    
    # Find first index with lower pressure than P_ref
    ind = 0
    for i in range(P_.shape[0]):
        if P_[i] < P_ref:
            ind = i
            break

    # Integrate from P_ref to TOA
    out2 = integrate.solve_ivp(hydrostatic_equation, [P_ref, P_[-1]], np.array([0.0]), t_eval=P_[ind:], args=args, rtol=1e-6)
    assert out2.success
    # Integrate from P_ref to BOA
    out1 = integrate.solve_ivp(hydrostatic_equation, [P_ref, P_[0]], np.array([0.0]), t_eval=P_[:ind][::-1], args=args, rtol=1e-6)
    assert out1.success

    # Stitch together
    z_ = np.append(out1.y[0][::-1],out2.y[0])

    return P_, T_, mubar_, z_

def equilibrium_temperature(stellar_radiation, bond_albedo):
    T_eq = ((stellar_radiation*(1.0 - bond_albedo))/(4.0*const.sigma))**(0.25)
    return T_eq 

def bolometric_flux(Teq, bond_albedo):
    stellar_radiation = 4.0*const.sigma*Teq**4/(1.0 - bond_albedo)
    return stellar_radiation

def _bolometric_flux(wv, F):
    return 1e-3*np.sum(F[:-1]*(wv[1:]-wv[:-1]))

def write_photochem_spectrum(wv, F, bolometric_radiation, Teff, outfile, plot=True):
    """Generates a UV flux file compatable with Photochem

    Parameters
    ----------
    wv : ndarray
        Array of wavelengths in nm
    F : ndarray
        Array of stellar fluxes in mW/m^2/nm
    bolometric_radiation : float
        The total irradiation at a planet in W/m^2
    Teff : float
        The stellar effective temperature
    outfile : str
        Output file name
    plot : bool, optional
        If True, then a plot is made, by default True
    """    
    if wv[-1]/1e3 < 100.0:
        # Add a blackbody spectrum to 100 microns
        wvb_cm = np.linspace((wv[-1]+wv[-1]*1e-5)/1e3,100,1000)/1e4
        wvb_nm = wvb_cm*(1/1e2)*(1e9/1) 
        Fb = fluxes.blackbody(Teff, wvb_cm)[0]*(1e2/1)*(1/1e9) # convert to mW/m^2/nm

        # Scale the blackbody, so it fits with the original spectrum
        factor = F[-1]/Fb[0]
        Fb = Fb*factor

        # Stitch together spectrum and blackbody
        wv_new = np.append(wv,wvb_nm)
        F_new = np.append(F,Fb)
    else:
        wv_new = wv
        F_new = F

    # Scale to match solar constant
    stellar_constant_tmp = _bolometric_flux(wv_new, F_new)
    factor = bolometric_radiation/stellar_constant_tmp
    F_new = F_new*factor

    # Write the output file
    fmt = '{:30}'
    with open(outfile,'w') as f:
        f.write(fmt.format('Wavelength (nm)'))
        f.write(fmt.format('Solar flux (mW/m^2/nm)'))
        f.write('\n')
        for i in range(F_new.shape[0]):
            f.write(fmt.format('%e'%wv_new[i]))
            f.write(fmt.format('%e'%F_new[i]))
            f.write('\n')

    if plot:
        # Plot
        plt.rcParams.update({'font.size': 15})
        fig,ax = plt.subplots(1,1,figsize=[6,5])
        fig.patch.set_facecolor("w")

        # Plot the spectrum
        ax.plot(wv_new,F_new,lw=.5, c='k', label='Spectrum')

        # Plot blackbody spectrum
        wvb_cm = np.linspace(.1,100,10000)/1e4
        wvb_nm = wvb_cm*(1/1e2)*(1e9/1) 
        Fb = fluxes.blackbody(Teff, wvb_cm)[0]*(1e2/1)*(1/1e9) # convert to mW/m^2/nm
        stellar_constant_b = _bolometric_flux(wvb_nm, Fb)
        factor = bolometric_radiation/stellar_constant_b
        Fb = Fb*factor
        ax.plot(wvb_nm,Fb,alpha=0.4,label='T = %i K blackbody'%(Teff))

        stellar_constant = _bolometric_flux(wv_new, F_new)
        T_eq = equilibrium_temperature(stellar_constant, 0.0)
        note = 'Bolometric flux = %.1f W/m$^2$\n'%(stellar_constant) \
            +'Equilibrium Temperature = %i K'%(T_eq)
        ax.text(.97, .02, note, \
                size = 11, ha='right', va='bottom',transform=ax.transAxes)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(1e1,1e5)
        ax.set_ylim(1e-5,2e3)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Stellar flux (mW m$^{-2}$ nm$^{-1}$)')
        ax.grid(alpha=0.8)
        ax.legend(ncol=1,bbox_to_anchor=(0.02,1.05),loc='lower left',fontsize=13)
        plt.show()
