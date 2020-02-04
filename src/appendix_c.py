import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

class StellarCore:
    """
    Variables that is to stay fixed at all times.
    """
    avg_rho_sol = 1.408e3           # Average density of the sun [kg/m^3]
    L0          = 3.828e26          # Luminosity [W]
    M0          = 0.8*1.989e30      # Mass [kg]
    
    X           = 0.7               # Hydrogen-1
    Y           = 0.29              # Helium-3
    Y_3         = 1e-10             # Helium-4
    Z           = 0.1               # "Metals"
    Z_73        = 1e-13             # Lithium-7
    Z_74        = 1e-13             # Berrylium-7
    
    my          = 1/(2*X + Y_3 + 3*Y/4 + 4*Z_73/7 + 5*Z_74/7)
    kB          = 1.38064852e-23
    
    def __init__(self):
        """
        Variables for the stellar core, and fraction of the particles it
        contains.
        """
        self.R0   = 0.72*6.95508e8         # Radius [m]
        self.rho0 = 5.1*self.avg_rho_sol   # Density [kg/m^3]
        self.T0   = 5.7e6                  # Temperature [K]
        self.P0   = 5.2e14
            
    def rho(self, P, T):
        mu = 1.66053904e-27
        return self.my*mu*P/(self.kB*T)
    
    def pressure(self, rho, T):
        mu = 1.66053904e-27
        return rho/(self.my*mu)*self.kB*T
    
    def readfile(file):
        """
            Making three lists with the 10-logarithm of the R-, T- and K-parameters
        from the text file 'opacity.txt' respectively, where R = rho/(T/10^6)^3
        comes in cgs units [g/cm^3]; T comes in SI units [K]; and K comes in cgs
        units [cm^2/g].
            Returning a function that takes in the 10-logarithm of R- and T- para-
        meters respectively, and returns the interpolated value of the 10-logarithm
        of the K-parameter.
        """
        infile = open(file)
        
        log10_R = list(map(float, infile.readline().split()[1:]))
        log10_T = np.zeros(70)
        log10_K = np.zeros((70,19))
        
        infile.readline()
        
        for i,line in enumerate(infile):
            log10_T[i] = line.split()[0]
            log10_K[i] = line.split()[1:]
            
        log_kappa = interp2d(log10_R, log10_T, log10_K)
        
        return log_kappa

def get_kappa(T, rho):
    """
    Function that takes T- and rho-values, and returns kappa in SI-units.
    """
    log_kappa = readfile('opacity.txt')
    
    log10_T = np.log10(T)
    log10_R = np.log10(rho/(T/1e6)**3)          # R = rho/(T/10^6)^3
    
    for i in range(len(T)):
        kappa = 10**(log_kappa(log10_R[i], log10_T[i]))
        kappa = float(kappa*(1e3/1e4))
        print('{:8.3f} {:9.2f} {:9.2f} {:9.2e}'.format(log10_T[i], log10_R[i],\
                                                  float(log_kappa(log10_R[i], log10_T[i])), kappa))

def sanity_interpolation():
    """
    Sanity check for interpolation of the function made for the opacity table.
    """
    log_kappa = readfile('opacity.txt')
    
    log10_R = np.array((-6.00, -5.95, -5.80, -5.70, -5.55, -5.95, -5.95, -5.95,\
                        -5.80, -5.75, -5.70, -5.55, -5.50))
    log10_T = np.array((3.750, 3.755, 3.755, 3.755, 3.755, 3.770, 3.780, 3.795,\
                        3.770, 3.775, 3.780, 3.795, 3.800))
    
    print('  log10_T   log10_R   log10_K   K(SI)')
    print('---------------------------------------')
    for i in range(len(log10_T)):
        kappa = 10**(log_kappa(log10_R[i], log10_T[i]))
        kappa = float(kappa*(1e3/1e4))
        print('{:8.3f} {:9.2f} {:9.2f} {:9.2e}'.format(log10_T[i], log10_R[i],\
                                                  float(log_kappa(log10_R[i], log10_T[i])), kappa))
    

# =============================================================================

# Some constants:

MeV_J  = 1.60217662e-19*1e6             # Converting from MeV to J
mu     = 1.66053904e-27                 # Atomic mass constant [kg]
N_A    = 6.022e23                       # Avogadros number

# Sun's variables:

rho       = 1.62e5                      # kg/m^3
temp_core = 1.57e7                      # K
temp_2    = 1e8                         # K

# Fraction of particles:

X      = 0.7                            # Hydrogen
Y3     = 1e-10                          # Helium-3
Y      = 0.29                           # Helium-4
Z      = 0.01                           # "Metals"
Z_73   = 1e-7                           # Lithium
Z_74   = 1e-7                           # Berrylium

# Densities of different particles:
# (H-1, He-3, He-4, e, Li-7, Be-7)
my = 1/(X + Y/4 + Z/7)

n_densities    = np.array((rho*X/mu, rho*Y3/(3*mu), rho*Y/(4*mu), 0,\
                        rho*Z_73/(7*mu), rho*Z_74/(7*mu)))

n_densities[3] = n_densities[0] + 2*n_densities[1] + 2*n_densities[2]\
                 + 1.5*n_densities[4] + 2*n_densities[5]       # Electron density

# =============================================================================

def prop_func(T):
    
    T_9    = T/1e9
    T_9_34 = T_9/(1 + 4.95e-2*T_9)
    T_9_17 = T_9/(1 + 0.759*T_9)

    l_pp  = (4.01e-15*T_9**(-2/3)*np.exp(-3.380*T_9**(-1/3))*(1 + 0.123*T_9**(1/3) +\
            1.09*T_9**(2/3) + 0.938*T_9))/(N_A*1e6)
    
    l_33  = (6.04e10*T_9**(-2/3)*np.exp(-12.276*T_9**(-1/3))*(1 + 0.034*T_9**(1/3)\
            - 0.522*T_9**(2/3) - 0.124*T_9 + 0.353*T_9**(4/3) + 0.213*T_9**(-5/3)))/(N_A*1e6)
    
    l_34  = (5.61e6*T_9_34**(5/6)*T_9**(-3/2)*np.exp(-12.826*T_9_34**(-1/3)))/(N_A*1e6)
    
    l_7e  = (1.34e-10*T_9**(-1/2)*(1 - 0.537*T_9**(1/3) + 3.86*T_9**(2/3) + 0.0027*T_9**(-1)\
            *np.exp(2.515e-3*T_9**(-1))))/(N_A*1e6)
    
    l_71_ = (1.096e9*T_9**(-2/3)*np.exp(-8.427*T_9**(-1/3)) - 4.830e8*T_9_17**(5/6)*T_9**(-3/2)\
             *np.exp(-8.472*T_9_17**(-1/3)) + 1.06e10*T_9**(-3/2)*np.exp(-30.442*T_9**(-1)))/(N_A*1e6)
    
    l_71  = (3.11e5*T_9**(-2/3)*np.exp(-10.262*T_9**(-1/3))\
             + 2.53e3*T_9**(-3/2)*np.exp(-7.306*T_9**(-1)))/(N_A*1e6)
    
    lambdas = np.array((l_pp, l_33, l_34, l_7e, l_71_, l_71))
    
    return lambdas

# Energy outcome:

Q_positron = 1.02
Q_init     = 0.15 + Q_positron + 5.49
Q_PPI      = 12.86
Q_PPII_1   = 1.59
Q_PPII_2   = 0.05           # neutrino flies away with most of the energy
Q_PPII_3   = 17.35
Q_PPIII_1  = Q_PPII_1
Q_PPIII_2  = 0.14
Q_PPIII_3  = Q_positron + 6.88
Q_PPIII_4  = 3.00

Q = np.array((Q_init, Q_PPI, Q_PPII_1, Q_PPII_2, Q_PPII_3, Q_PPIII_1, Q_PPIII_2,\
              Q_PPIII_3, Q_PPIII_4))*MeV_J  # In unit J

# =============================================================================

def reaction(T):
    
    def not_adjusted(ni, nk):
        if ni == nk:
            delta = 1
            
        else:
            delta = 0
                
        return n_densities[ni]*n_densities[nk]/(rho*(1 + delta))
    
    lambdas = prop_func(T)
    
    r = np.zeros(6)
    r_pp  = not_adjusted(0, 0)*lambdas[0]
    r_33  = not_adjusted(1, 1)*lambdas[1]
    r_34  = not_adjusted(1, 2)*lambdas[2]
    r_7e  = not_adjusted(5, 3)*lambdas[3]
    r_71_ = not_adjusted(4, 0)*lambdas[4]
    r_71  = not_adjusted(5, 0)*lambdas[5]

    r = np.array([r_pp, r_33, r_34, r_7e, r_71_, r_71])
    
    if r[0] < 2*r[1] + r[2]:
        K = r[0]/(2*r[1] + r[2])
        r[1] = K*r[1]
        r[2] = K*r[2]
        
    if r[2] < r[3] + r[5]:
        K = r[2]/(r[3] + r[5])
        r[3] = K*r[3]
        r[5] = K*r[5]
        
    if r[3] < r[4]:
        r[4] = r[3]
            
    return r

def sanity_test(T):
    r_Q = reaction(T)

    r_Q *= rho

    r_Q[0] *= Q[0]
    r_Q[1] *= Q[1]
    r_Q[2] *= Q[2]
    r_Q[3] *= Q[3]
    r_Q[4] *= Q[4]
    r_Q[5] *= (Q[6] + Q[7] + Q[8])
    
    print('Energy production:')
    print('Temp = %.2e K' %T)
    print('-------------------------------------')
    for i in range(len(r_Q)):
        print('{:.2e}'.format(r_Q[i]))

if __name__ == '__main__':
    #sanity_test(temp_2)
    #readfile('opacity.txt')
    #sanity_interpolation()
    #get_kappa(T_test,rho_test)
    a = StellarCore()
    print('{:e}'.format(a.pressure(a.rho(a.P0, a.T0), a.T0)))