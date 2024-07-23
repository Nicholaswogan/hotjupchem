import numpy as np
from matplotlib import pyplot as plt
from astropy import constants
import pickle
from hotjupchem import EvoAtmosphereHJ, zahnle_earth_HHeCNOS, zahnle_earth_thermo
from hotjupchem.climate import ClimateHJ, make_outfile_name

def make_plot(sol, soleq):
    plt.rcParams.update({'font.size': 15})
    fig,ax = plt.subplots(1,1,figsize=[6,5])
    fig.patch.set_facecolor("w")

    species = ['CO2','H2S','SO2','S','S2','SO','CH4']
    names = ['CO$_2$','H$_2$S','SO$_2$','S','S$_2$','SO','CH$_4$']
    colors = ['C0','C1','violet','gold','grey','skyblue','k']
    for i,sp in enumerate(species):
        ax.plot(sol[sp],1e3*sol['pressure']/1e6,label=names[i],c=colors[i], lw=2.5, alpha=0.8)
        ax.plot(soleq[sp],1e3*soleq['pressure']/1e6,c=colors[i], lw=2.5, ls=':', alpha=0.4)

    ax.set_xlim(1e-8,1e-3)
    ax.set_ylim(1e4,5e-4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(alpha=0.4)
    ax.legend(ncol=3,bbox_to_anchor=(0.02,1.05),loc='lower left')
    ax.set_xlabel('Mixing Ratio')
    ax.set_ylabel('Pressure (mbar)')

    plt.savefig('WASP39b_Fig1_Tsai2023.pdf',bbox_inches='tight')

def main():
    # If true, then runs climate model.
    run_climate = False

    planet_mass = 5.314748872540942e+29 # grams
    planet_radius = 9079484000.0 # cm
    P_ref = 1e6 # dynes/cm^3
    metallicity = 10.0 # planet metallicity
    CtoO = 1.0 # C/O ratio
    tint = 200 # tint

    # Run climate
    if run_climate:
        planet_mass_ = planet_mass/(constants.M_earth.value*1e3)
        planet_radius_ = planet_radius/(constants.R_earth.value*1e2)
        semi_major = 0.04828 # AU
        Teq = 1120.55 # K
        T_star = 5400 # K
        logg_star = 4.45
        metal_star = 0.01 # log10
        r_star = 0.932 # solar radii
        database_dir = '/Users/nicholas/Applications/picaso_data/climate/'

        # Iniitalize climate model
        c = ClimateHJ(planet_mass_, planet_radius_, P_ref, semi_major, Teq, 
        T_star, logg_star, metal_star, r_star, database_dir)

        out = c.run_climate_model(metallicity, CtoO, tint, save_output=True)
    else:
        with open(make_outfile_name(np.log10(metallicity), CtoO, tint),'rb') as f:
            out = pickle.load(f)

    # Initialize photochem model
    pc = EvoAtmosphereHJ(zahnle_earth_HHeCNOS,'WASP39_flux.txt',planet_mass, planet_radius, 
                         nz=100, thermo_file=zahnle_earth_thermo)

    # Prepare photochem inputs
    P = out['pressure'][::-1].copy()*1e6
    T = out['temperature'][::-1].copy()
    Kzz = np.ones(P.shape[0])*1e7
    for i in range(P.shape[0]):
        if P[i]/1e6 > 5.0:
            Kzz[i] = 5e7
        else:
            Kzz[i] = 5e7*(5/(P[i]/1e6))**0.5
    pc.initialize_to_climate_equilibrium_PT(P, T, Kzz, metallicity, CtoO)

    # Photochemical equilibrium
    success = pc.find_steady_state()

    # get output
    sol = pc.return_atmosphere()
    soleq = pc.return_atmosphere(equilibrium=True)

    # Plot
    make_plot(sol, soleq)

if __name__ == '__main__':
    main()


