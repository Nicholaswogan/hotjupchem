import numpy as np
from picaso import justdoit as jdi
from astropy import constants
import astropy.units as u
import pickle
from . import utils

def make_outfile_name(mh, CtoO, tint):
    outfile = 'MH=%.3f_CO=%.3f_Tint=%.1f.pkl'%(mh, CtoO, tint)
    return outfile

class ClimateHJ():

    def __init__(self, planet_mass, planet_radius, P_ref, semi_major, Teq, 
                 T_star, logg_star, metal_star, r_star, database_dir):
        """Initialized the climate model.

        Parameters
        ----------
        planet_mass : float
            Planet mass in Earth masses
        planet_radius : float
            Plane radius in Earth radii
        P_ref : float
            Reference pressure in dynes/cm^2
        semi_major : float
            Semi-major axis in AU
        Teq : float
            Equilibrium temperature in K
        T_star : float
            Stellar effective temperature in K
        logg_star : float
            Stellar gravity in logg
        metal_star : float
            Stellar metallicity in log10 units
        r_star : float
            Stellar radius in solar radii
        database_dir : str
            Path to where climate opacities are stored.
        """        
        self.nlevel = 91
        self.nofczns = 1
        self.nstr_upper = 85
        self.rfacv = 0.5 
        self.p_bottom = 3 # log10(bars)
        self.planet_mass = planet_mass # Earth masses
        self.planet_radius = planet_radius # Earth radii
        self.P_ref = P_ref # dynes/cm^2
        self.semi_major = semi_major # AU
        self.Teq = Teq # K
        self.T_star = T_star # K
        self.logg_star = logg_star
        self.metal_star = metal_star # log10 metallicity
        self.r_star = r_star # solar radii
        self.database_dir = database_dir
        self.outfolder = './'

    def run_climate_model(self, metallicity, CtoO, tint, save_output=False):
        """Runs the climate model.

        Parameters
        ----------
        metallicity : float
            Metallicity relative to solar
        CtoO : float
            The C/O ratio relative to solar
        tint : float
            Intrinsic temperature
        save_output : bool, optional
            If True, the the output is saved to a pickle file, by default False

        Returns
        -------
        dict
            Dictionary containing the P-T profile
        """        
        # Get the opacity database
        mh = np.log10(metallicity)
        if mh >= 0:   
            mh_str = ('+%.2f'%mh).replace('.','')
        else:
            mh_str = ('-%.2f'%mh).replace('.','')
        CtoO_str = ('%.2f'%CtoO).replace('.','')

        ck_db = self.database_dir+f'sonora_2020_feh{mh_str}_co_{CtoO_str}.data.196'
        opacity_ck = jdi.opannection(ck_db=ck_db)
        
        # Initialize climate run
        cl_run = jdi.inputs(calculation="planet", climate = True)

        cl_run.inputs['approx']['p_reference'] = self.P_ref/1e6

        # set gravity
        grav = utils.gravity(self.planet_radius*constants.R_earth.value*1e2, self.planet_mass*constants.M_earth.value*1e3, 0.0)/1e2
        cl_run.gravity(gravity=grav, gravity_unit=u.Unit('m/(s**2)'))
        
        # Set tint
        cl_run.effective_temp(tint) 

        # Set stellar properties
        T_star = self.T_star
        logg = self.logg_star #logg, cgs
        metal = self.metal_star # metallicity of star
        r_star = self.r_star # solar radius
        semi_major = self.semi_major # star planet distance, AU
        cl_run.star(opacity_ck, temp =T_star,metal =metal, logg =logg, radius = r_star,
                    radius_unit=u.R_sun,semi_major= semi_major , semi_major_unit = u.AU, database='phoenix')

        # Initial temperature guess
        nlevel = self.nlevel # number of plane-parallel levels in your code
        Teq = self.Teq # planet equilibrium temperature
        pt = cl_run.guillot_pt(Teq, nlevel=nlevel, T_int = tint, p_bottom=self.p_bottom, p_top=-6)
        temp_guess = pt['temperature'].values
        pressure = pt['pressure'].values

        nofczns = self.nofczns # number of convective zones initially. Let's not play with this for now.
        nstr_upper = self.nstr_upper # top most level of guessed convective zone
        nstr_deep = nlevel - 2 # this is always the case. Dont change this
        nstr = np.array([0,nstr_upper,nstr_deep,0,0,0]) # initial guess of convective zones
        rfacv = self.rfacv

        # Set inputs
        cl_run.inputs_climate(temp_guess= temp_guess, pressure= pressure,
                        nstr = nstr, nofczns = nofczns , rfacv = rfacv)

        # Run model
        out = cl_run.climate(opacity_ck)

        if save_output:
            outfile = self.outfolder+make_outfile_name(mh, CtoO, tint)
            with open(outfile,'wb') as f:
                pickle.dump(out,f)

        return out