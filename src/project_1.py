import numpy as np
import matplotlib.pyplot as plt
import constants as c
from scipy.interpolate import interp2d

class StellarCore:
    """
    Variables that is to stay fixed at all times, and some constants.
    """
        
    avg_rho_sol = 1.408e3           # Average density of the sun [kg/m^3]
    L0          = 3.828e26          # Luminosity [W]
    
    X           = 0.7               # Hydrogen-1
    Y           = 0.29              # Helium-3
    Y_3         = 1e-10             # Helium-4
    Z           = 0.01              # "Metals"
    Z_73        = 1e-13             # Lithium-7
    Z_74        = 1e-13             # Berrylium-7
    
    kB    = 1.38064852e-23          # m^2 kg/s^2 K
    G     = 6.67408e-11             # m^3/kg s^2
    MeV_J = 1.60217662e-19*1e6      # Converting from MeV to J
    mu    = 1.66053904e-27          # Atomic mass constant [kg]
    N_A   = 6.022e23                # Avogadros number
    my    = 1/(2*X + Y_3 + 3*Y/4 + 4*Z_73/7 + 5*Z_74/7)   # average molecular weight
    sigma = 5.67e-8                 # W/m^2 K^4
    a     = 4*sigma/c.c             # J/m^3 K^4
    
    def __init__(self, R0=0.72*c.R_sol, rho0=5.1*1.408e3, T0=5.7e6, P0=5.2e14, M0=0.8*c.M_sol):
        """
        Initial parameters for the radiative zone in the sun
        """
        self.R0   = R0                      # Radius [m]
        self.rho0 = rho0                    # Density [kg/m^3]
        self.T0   = T0                      # Temperature [K]
        self.P0   = P0                      # Pressure [Pa]
        self.M0   = M0                      # Mass [kg]
    
    def readfile(self, file):
        """
            Making three lists with the 10-logarithm of the R-, T- and K-parameters
        from the text file 'opacity.txt' respectively, where R = rho/(T/10^6)^3
        comes in cgs units [g/cm^3]; T comes in SI units [K]; and K (kappa) comes
        in cgs units [cm^2/g].
            Then returning a function that takes in the 10-logarithm of R- and 
        T-parameters respectively, and returns the interpolated value of the
        10-logarithm of the K-parameter.
        """
        infile = open(file)
        
        log10_R = list(map(float, infile.readline().split()[1:]))
        log10_T = np.zeros(70)
        log10_K = np.zeros((70,19))
        
        infile.readline()
        
        for i,line in enumerate(infile):
            log10_T[i] = line.split()[0]
            log10_K[i] = line.split()[1:]
            
        self.log_kappa = interp2d(log10_R, log10_T, log10_K)
    
    def lambd_ik(self, T):
        """
            Calculating the reaction rate in units [cm^3/s] for the different
        steps in the proton-proton chain reaction. The first lambda (l_pp)
        returns the reaction rate for both of the setps for merging deuterium
        and helium.
        """
        
        T_9    = T/1e9
        T_9_34 = T_9/(1 + 4.95e-2*T_9)
        T_9_17 = T_9/(1 + 0.759*T_9)
    
        l_pp  = (4.01e-15*T_9**(-2/3)*np.exp(-3.380*T_9**(-1/3))\
                 *(1 + 0.123*T_9**(1/3) + 1.09*T_9**(2/3) + 0.938*T_9))\
                /(self.N_A*1e6)
        
        l_33  = (6.04e10*T_9**(-2/3)*np.exp(-12.276*T_9**(-1/3))\
                 *(1 + 0.034*T_9**(1/3) - 0.522*T_9**(2/3)\
                   - 0.124*T_9 + 0.353*T_9**(4/3) + 0.213*T_9**(-5/3)))\
                /(self.N_A*1e6)
        
        l_34  = (5.61e6*T_9_34**(5/6)*T_9**(-3/2)*np.exp(-12.826*T_9_34**(-1/3)))\
                /(self.N_A*1e6)
        
        l_7e  = (1.34e-10*T_9**(-1/2)*(1 - 0.537*T_9**(1/3) + 3.86*T_9**(2/3)\
                 + 0.0027*T_9**(-1)*np.exp(2.515e-3*T_9**(-1))))/(self.N_A*1e6)
        
        l_71_ = (1.096e9*T_9**(-2/3)*np.exp(-8.427*T_9**(-1/3))\
                 - 4.830e8*T_9_17**(5/6)*T_9**(-3/2)*np.exp(-8.472*T_9_17**(-1/3))\
                 + 1.06e10*T_9**(-3/2)*np.exp(-30.442*T_9**(-1)))\
                /(self.N_A*1e6)
        
        l_71  = (3.11e5*T_9**(-2/3)*np.exp(-10.262*T_9**(-1/3))\
                 + 2.53e3*T_9**(-3/2)*np.exp(-7.306*T_9**(-1)))\
                /(self.N_A*1e6)
        
        lambdas = np.array([l_pp, l_33, l_34, l_7e, l_71_, l_71])
        
        return lambdas
    
    def epsilon(self, T, rho):
        """
            Calculating the total energy generation per unit mass.
            Starting by determining the reaction rate with respect to mass per
        step in the PP-chain, and then multiplying each step's reaction rate
        with its corresponding energy output.
        """
        mu = self.mu
        
        n_densities    = np.array((rho*self.X/self.mu, rho*self.Y_3/(3*mu),\
                                   rho*self.Y/(4*mu), 0, rho*self.Z_73/(7*mu),\
                                   rho*self.Z_74/(7*mu)))

        n_densities[3] = n_densities[0] + 2*n_densities[1] + 2*n_densities[2]\
                         + 1.5*n_densities[4] + 2*n_densities[5]       # Electron density
        
        def not_adjusted(ni, nk):
            """
            Implementing the delta function in the reaction rate.
            """
            if ni == nk:
                delta = 1
                
            else:
                delta = 0
                    
            return n_densities[ni]*n_densities[nk]/(rho*(1 + delta))
        
        lambdas = self.lambd_ik(T)
        
        # Reaction rates per step in the PP-chain
        
        r = np.zeros(6)
        r_pp  = not_adjusted(0, 0)*lambdas[0]
        r_33  = not_adjusted(1, 1)*lambdas[1]
        r_34  = not_adjusted(1, 2)*lambdas[2]
        r_7e  = not_adjusted(5, 3)*lambdas[3]
        r_71_ = not_adjusted(4, 0)*lambdas[4]
        r_71  = not_adjusted(5, 0)*lambdas[5]
    
        r = np.array([r_pp, r_33, r_34, r_7e, r_71_, r_71])
        
        # Checking if the next step's reaction rate is bigger than the foregoing
        
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
        
        # Energy outputs per step in PP-chain
            
        Q_positron = 1.02                       # MeV
        Q_init     = 0.15 + Q_positron + 5.49   # MeV
        Q_PPI      = 12.86                      # MeV
        Q_PPII_1   = 1.59                       # MeV
        Q_PPII_2   = 0.05                       # MeV
        Q_PPII_3   = 17.35                      # MeV
        Q_PPIII_1  = Q_PPII_1                   # MeV
        Q_PPIII_2  = 0.14                       # MeV
        Q_PPIII_3  = Q_positron + 6.88          # MeV
        Q_PPIII_4  = 3.00                       # MeV
        
        Q = np.array((Q_init, Q_PPI, Q_PPII_1, Q_PPII_2, Q_PPII_3, Q_PPIII_1, Q_PPIII_2,\
                      Q_PPIII_3, Q_PPIII_4))*self.MeV_J  # J
        
        r_Q = r
    
        r_Q[0] *= Q[0]                          # W/s
        r_Q[1] *= Q[1]                          # W/s
        r_Q[2] *= Q[2]                          # W/s
        r_Q[3] *= Q[3]                          # W/s
        r_Q[4] *= Q[4]                          # W/s
        r_Q[5] *= (Q[6] + Q[7] + Q[8])          # W/s
        
        return r_Q                              # W/s
    
    def rho(self, P, T):
        return self.my*self.mu*P/(self.kB*T)    # kg/m^3
    
    def pressure(self, rho, T):
        return rho/(self.my*self.mu)*self.kB*T  # Pa

    def get_kappa(self, T, rho):
        """
        Function that takes T- and rho-values, and returns kappa in SI-units.
        """
        
        if not type(T) == list:
            """
            Converting type T into list if type T is e.g. a scalar.
            """
            T = np.array([T])
        
        log10_T = np.log10(T)
        log10_R = np.log10(rho/((T/1e6)**3)*1e3/1e2**3)    # R = rho/(T/10^6)^3 [cgs]
        
        for i in range(len(T)):
            kappa = 10**(self.log_kappa(log10_R[i], log10_T[i]))
            kappa = float(kappa*(1e3/1e4))      # Converting from cgs to SI units
            
        return kappa                            # m^2/kg
    
    def solver(self, mass, initials, find=False, exp=False, dm=1e-4, dynamic=True):
        """
        Solving the four partial differential equations respectively as they're
        given in the projct text using the Euler method. 
        """
        
        N = 10000
        p = 0.001
        
        R   = np.zeros(N)                       # m
        P   = np.zeros(N)                       # Pa
        L   = np.zeros(N)                       # W
        T   = np.zeros(N)                       # K
        rho = np.zeros(N)                       # kg/m^3
        M   = np.zeros(N)                       # kg
        
        L[0]   = self.L0
        M[0]   = mass
        R[0]   = initials[0]
        T[0]   = initials[1]
        rho[0] = initials[2]
        P_rad  = self.a/3*T[0]**4
        P[0]   = self.pressure(rho[0], T[0]) + P_rad
        
        dm = -abs(dm*self.M0)
        
        for i in range(len(M)-1):
            
            f_1 = 1/(4*np.pi*R[i]**2*rho[i])                            # dR/dM
            f_2 = -self.G*M[i]/(4*np.pi*R[i]**4)                        # dP/dM
            f_3 = np.sum(self.epsilon(T[i], rho[i]))                    # dL/dM
            f_4 = -(3*self.get_kappa(T[i], rho[i])*L[i]/\
                            (256*np.pi**2*self.sigma*R[i]**4*T[i]**3))  # dT/dM
            
            if dynamic:
                dm = [p*R[i]/f_1, p*P[i]/f_2, p*L[i]/f_3, p*T[i]/f_4]
                dm = -np.amin(np.abs(dm))
            
            R[i+1]   = R[i] + f_1*dm
            P[i+1]   = P[i] + f_2*dm
            L[i+1]   = L[i] + f_3*dm
            T[i+1]   = T[i] + f_4*dm
            rho[i+1] = self.rho(P[i+1], T[i+1])
            M[i+1]   = M[i] + dm
            
            if M[i+1] < 0:
                # Breaking the loop if the mass becomes negative
                
                R[i+1] = P[i+1] = L[i+1] = T[i+1] = rho[i+1] = P[i+1] = M[i+1] = 0
                print('Mass < 0 at iteration {:d} of {:d}'\
                      .format(i, N))
                break
        
        if exp or find:
            return R, P, L, T, rho, M
        
        else:
            return R[:i+1], P[:i+1], L[:i+1], T[:i+1], rho[:i+1], M[:i+1]
        
    def plot(self, mass, initials, find=False, exp=False, test_R=False):
        """
        Making the different plots depending of what's true in the arguments to
        self.plot().
        """
        
        if test_R:
            """
            Testing which effects significantly different step sizes dm has for
            the radius.
            """
            dm = np.array((1, 1e-1, 1e-2, 1e-3, 1e-4))
        
            for i in range(len(dm)):
                R, P, L, T, rho, M = self.solver(mass, initials, dynamic=False, dm=dm[i])
                
                M0  = self.M0
                rho = self.rho(P, T)
                
                plt.plot(M/M0, R/self.R0)
                plt.title('Radius vs mass with dm={:.1e}'.format(dm[i]))
                plt.xlabel(r'$M/M_{\odot}$')
                plt.ylabel(r'$R/R_{\odot}$')
                
                plt.show()
            
            exit()
        
        elif exp:
            """
            Plotting the values (R(M), P(M), L(M), T(M) and rho(M)) from the
            function self.experiments, changing only one of the parameters R0,
            T0 and rho0 respectively per plot.
            """
            R, P, L, T, rho, M = self.experiments(mass)
            P0 = self.pressure(self.rho0, self.T0)
            M0 = self.M0

            l    = self.l
            indx = 0
            
            for j in range(3):
                for i in range(l):
                    
                    plt.subplot(3,2,1)
                    plt.plot(np.trim_zeros(M[j+(l-1)*j+i])/M0,\
                             np.trim_zeros(R[j+(l-1)*j+i])/self.R0)
                    plt.title('R(M)', fontsize=19)
                    plt.ylabel(r'$R/R_{0}$', fontsize=19)
                    plt.xticks(fontsize=17)
                    plt.yticks(fontsize=17)
                    
                    ax2 = plt.subplot(3,2,4)
                    plt.semilogy(np.trim_zeros(M[j+(l-1)*j+i])/M0,\
                                 np.trim_zeros(rho[j+(l-1)*j+i])/self.rho0)
                    plt.title(r'$\rho(M)$', fontsize=19)
                    plt.ylabel(r'$\rho/\rho_{0}$', fontsize=19)
                    ax2.yaxis.tick_right()
                    ax2.yaxis.set_label_position('right')
                    plt.xticks(fontsize=17)
                    plt.yticks(fontsize=17)
                    
                    ax1 = plt.subplot(3,2,2)
                    plt.plot(np.trim_zeros(M[j+(l-1)*j+i])/M0,\
                             np.trim_zeros(L[j+(l-1)*j+i])/self.L0)
                    plt.title('L(M)', fontsize=19)
                    plt.ylabel(r'$L/L_{0}$', fontsize=19)
                    ax1.yaxis.tick_right()
                    ax1.yaxis.set_label_position('right')
                    plt.xticks(fontsize=17)
                    plt.yticks(fontsize=17)
                    
                    plt.subplot(3,2,3)
                    plt.plot(np.trim_zeros(M[j+(l-1)*j+i])/M0,\
                             np.trim_zeros(T[j+(l-1)*j+i])/1e6)
                    plt.title('T(M)', fontsize=19)
                    plt.ylabel(r'$T[MK]$', fontsize=19)
                    plt.xticks(fontsize=17)
                    plt.yticks(fontsize=17)
                    
                    ax3 = plt.subplot(3,2,5)
                    plt.plot(np.trim_zeros(M[j+(l-1)*j+i])/M0,\
                             np.trim_zeros(P[j+(l-1)*j+i])/P0,\
                             label=r'${:.2f}R_0, {:.2f}T_0, {:.2}\rho_0$'\
                                   .format(R[indx,0]/self.R0, T[indx,0]/self.T0,\
                                           rho[indx,0]/self.rho0))
                    plt.title('P(M)', fontsize=19)
                    plt.ylabel(r'$P/P_{0}$', fontsize=19)
                    chartBox = ax3.get_position()
                    ax3.set_position([chartBox.x0, chartBox.y0, chartBox.width,\
                                     chartBox.height])
                    plt.xticks(fontsize=17)
                    plt.yticks(fontsize=17)
                    
                    indx += 1
                
                ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 0.9),\
                           fontsize=17, shadow=True, ncol=2)
                
                plt.show()
            exit()
            
        elif find:
            """
            Plotting the values from the function self.finding_star, where the
            different initial values are varying independent of each other.
            """
            R, P, L, T, rho, M = self.finding_star(mass)
            M0 = self.M0
            P0 = self.pressure(self.rho0, self.T0)
            l  = self.n
            
            indx = 0
            
            for i in range(l):
                for j in range(l):
                    for k in range(l):
            
                        plt.subplot(3,2,1)
                        plt.plot(np.trim_zeros(M[i+(l**2-1)*i+(l-1)*j+j+k])/M0,\
                                 np.trim_zeros(R[i+(l**2-1)*i+(l-1)*j+j+k])/self.R0)
                        plt.title('R(M)')
                        plt.ylabel(r'$R/R_{0}$')
                        
                        ax2 = plt.subplot(3,2,4)
                        plt.semilogy(np.trim_zeros(M[i+(l**2-1)*i+(l-1)*j+j+k])/M0,\
                                     np.trim_zeros(rho[i+(l**2-1)*i+(l-1)*j+j+k])/self.rho0)
                        plt.title(r'$\rho(M)$')
                        plt.ylabel(r'$\rho/\rho_{0}$')
                        ax2.yaxis.tick_right()
                        ax2.yaxis.set_label_position('right')
                        
                        ax1 = plt.subplot(3,2,2)
                        plt.plot(np.trim_zeros(M[i+(l**2-1)*i+(l-1)*j+j+k])/M0,\
                                 np.trim_zeros(L[i+(l**2-1)*i+(l-1)*j+j+k])/self.L0)
                        plt.title('L(M)')
                        plt.ylabel(r'$L/L_{0}$')
                        ax1.yaxis.tick_right()
                        ax1.yaxis.set_label_position('right')
                        
                        plt.subplot(3,2,3)
                        plt.plot(np.trim_zeros(M[i+(l**2-1)*i+(l-1)*j+j+k])/M0,\
                                 np.trim_zeros(T[i+(l**2-1)*i+(l-1)*j+j+k])/1e6)
                        plt.title('T(M)')
                        plt.ylabel(r'$T[MK]$')
                        
                        ax3 = plt.subplot(3,2,5)
                        plt.plot(np.trim_zeros(M[i+(l**2-1)*i+(l-1)*j+j+k])/M0,\
                                 np.trim_zeros(P[i+(l**2-1)*i+(l-1)*j+j+k])/P0,\
                                 label=r'${:.4f}R_0, {:.4f}T_0, {:.4}\rho_0$'\
                                       .format(R[indx,0]/self.R0, T[indx,0]/self.T0,\
                                               rho[indx,0]/self.rho0))
                        plt.title('P(M)')
                        plt.ylabel(r'$P/P_{0}$')
                        chartBox = ax3.get_position()
                        ax3.set_position([chartBox.x0, chartBox.y0, chartBox.width,\
                                         chartBox.height])
                        
                        indx += 1
                
                ax3.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8),\
                           shadow=True, ncol=3)
                
                plt.show()
            exit()
        
        """
        Plotting R(M), L(M), T(M), rho(M) and P(M) respectively where the R0-,
        T0- and rho0-values are given from the argument "initials".
        """
        
        R, P, L, T, rho, M = self.solver(mass, initials)
        
        M0   = self.M0
        R0   = self.R0
        T0   = self.T0
        rho0 = self.rho0
        
        plt.subplot(3,2,1)
        plt.plot(M/M0, R/self.R0)
        plt.title('R(M)', fontsize=19)
        plt.ylabel(r'$R/R_{0}$', fontsize=19)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        ax2 = plt.subplot(3,2,4)
        plt.semilogy(M/M0, rho/rho0)
        plt.title(r'$\rho(M)$', fontsize=19)
        plt.ylabel(r'$\rho/\rho_{0}$', fontsize=19)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        ax1 = plt.subplot(3,2,2)
        plt.plot(M/M0, L/self.L0)
        plt.title('L(M)', fontsize=19)
        plt.ylabel(r'$L/L_{0}$', fontsize=19)
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position('right')
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.subplot(3,2,3)
        plt.plot(M/M0, T/1e6)
        plt.title('T(M)', fontsize=19)
        plt.ylabel(r'$T[MK]$', fontsize=19)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        ax3 = plt.subplot(3,2,5)
        plt.plot(M/M0, P/self.P0, label=r'${:.5f}R_0, {:.4f}T_0, {:.2f}\rho_0$'\
                                         .format(R[0]/R0, T[0]/T0, rho[0]/rho0))
        plt.title('P(M)', fontsize=19)
        plt.ylabel(r'$P/P_{0}$', fontsize=19)
        chartBox = ax3.get_position()
        ax3.set_position([chartBox.x0, chartBox.y0, chartBox.width,\
                         chartBox.height])
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        ax3.legend(loc='upper center', bbox_to_anchor=(1.75, 0.6),\
                   fontsize=17, shadow=True, ncol=1)
        
        plt.show()
        
        """
        Plotting T(R), L(R), epsilon(R), rho(R), P(R) and M(R) respectively
        """
        
        eps = np.zeros(len(R))          # Array for storing the epsilon values
        
        for i in range(len(eps)):
            eps[i] = np.sum(self.epsilon(T[i], rho[i]))
        
        plt.subplot(3,2,1)
        plt.plot(R/R0, T/1e6)
        plt.title('T(R)', fontsize=19)
        plt.ylabel(r'$T [MK]$', fontsize=19)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        ax1 = plt.subplot(3,2,2)
        plt.plot(R/R0, L/self.L0)
        plt.plot(R/R0, 0.995*np.ones(len(R)), label=r'$0.995L_{0}$')
        plt.title('L(R)', fontsize=19)
        plt.ylabel(r'$L/L_{0}$', fontsize=19)
        plt.legend(loc='lower right', fontsize=17)
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position('right')
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.subplot(3,2,3)
        plt.semilogy(R/R0, eps)
        plt.title(r'$\epsilon(R)$', fontsize=19)
        plt.ylabel(r'$\epsilon$', fontsize=19)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        ax2 = plt.subplot(3,2,4)
        plt.semilogy(R/R0, rho/rho0)
        plt.title(r'$\rho(R)$', fontsize=19)
        plt.ylabel(r'$\rho/\rho_{0}$', fontsize=19)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.subplot(3,2,5)
        plt.semilogy(R/R0, P/self.P0, label=r'${:.5f}R_0, {:.4f}T_0, {:.2f}\rho_0$'\
                                         .format(R[0]/R0, T[0]/T0, rho[0]/rho0))
        plt.title('P(R)', fontsize=19)
        plt.ylabel(r'$P/P_{0}$', fontsize=19)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.legend(loc='best', fontsize=17)
        
        ax3 = plt.subplot(3,2,6)
        plt.plot(R/R0, M/M0)
        plt.title('M(R)', fontsize=19)
        plt.ylabel(r'$M/M_{0}$', fontsize=19)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position('right')
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
                
        plt.show()
    
    def experiments(self, mass):
        """
        Experimenting with different initial values for R0, T0 and rho0,
        variating only one parameter per loop over the factor values.
        """
        
        factors   = np.linspace(1/5, 1, 10)
        
        self.l = len(factors)
        l      = self.l
        
        R   = np.zeros((3*l, 10000))
        L   = np.zeros((3*l, 10000))
        T   = np.zeros((3*l, 10000))
        rho = np.zeros((3*l, 10000))
        P   = np.zeros((3*l, 10000))
        M   = np.zeros((3*l, 10000))
        
        for i in range(3):
            for j in range(len(factors)):
                # Starting with updating back to original initial conditions
                variables = np.array([self.R0, self.T0, self.rho0])
                
                variables[i] = factors[j]*variables[i]
                
                R[i+(l-1)*i+j], P[i+(l-1)*i+j], L[i+(l-1)*i+j], T[i+(l-1)*i+j],\
                rho[i+(l-1)*i+j], M[i+(l-1)*i+j] = self.solver(mass, variables,\
                                                               exp=True)
        
        return R, P, L, T, rho, M

    
    def finding_star(self, mass):
        """
        This function variates the different variables simultaneously to find
        the best initial conditions for making both the radius and the
        luminosity go to zero as the mass goes to zero.
        """
        initials = np.zeros(3)
        
        self.n = 3
        l      = self.n
        
        R_factors   = np.linspace(0.49194, 0.49196, l)
        T_factors   = np.linspace(0.8024, 0.8026, l)
        rho_factors = np.linspace(0.7799, 0.7801, l)
        
        # Making l^3*10000 matrices for the parameters we want to look at:
        
        R   = np.zeros((l**3, 10000))
        L   = np.zeros((l**3, 10000))
        T   = np.zeros((l**3, 10000))
        rho = np.zeros((l**3, 10000))
        P   = np.zeros((l**3, 10000))
        M   = np.zeros((l**3, 10000))
        
        for i in range(l):
            for j in range(l):
                for k in range(l):
                    initials[0] = R_factors[i]*self.R0
                    initials[1] = T_factors[j]*self.T0
                    initials[2] = rho_factors[k]*self.rho0
                    
                    R[i+(l**2-1)*i+(l-1)*j+j+k], P[i+(l**2-1)*i+(l-1)*j+j+k],\
                    L[i+(l**2-1)*i+(l-1)*j+j+k], T[i+(l**2-1)*i+(l-1)*j+j+k],\
                    rho[i+(l**2-1)*i+(l-1)*j+j+k], M[i+(l**2-1)*i+(l-1)*j+j+k]\
                    = self.solver(mass, initials, find=True)
        
        return R, P, L, T, rho, M
    
    def print_values(self, mass, initials):
        """
        Printing out the final values of the parameters after the integration
        is done.
        """
        R, P, L, T, rho, M = self.solver(mass, initials)
        values = np.array((R, P, L, T, M))
        
        for var in values:
            print(var[-1])
    
    def sanity_plot(self, mass, initials, debug=False):
        """
        Sanity check for the given initial parameters and a step length of
        10^(-4).
        """
        R, P, L, T, rho, M = self.solver(mass, initials)
        
        if debug:
            #print(R)
            print(P)
            #print(L)
            print(T)
            exit()
        
        M0  = mass
        rho = self.rho(P, T)
        
        plt.subplot(2,2,1)
        plt.plot(M/M0, R/self.R_sol)
        plt.title('Radius vs mass')
        plt.xlabel(r'$M/M_{\odot}$')
        plt.ylabel(r'$R/R_{\odot}$')
        
        plt.subplot(2,2,4)
        plt.semilogy(M/M0, rho/self.avg_rho_sol)
        plt.title('Density vs mass')
        plt.xlabel(r'$M/M_{\odot}$')
        plt.ylabel(r'$\rho/\rho_{\odot}$')
        plt.xlim(0, 1)
        plt.ylim(1, 10)
        
        plt.subplot(2,2,2)
        plt.plot(M/M0, L/self.L0)
        plt.title('Luminosity vs mass')
        plt.xlabel(r'$M/M_{\odot}$')
        plt.ylabel(r'$L/L_{\odot}$')
        
        plt.subplot(2,2,3)
        plt.plot(M/M0, T/1e6)
        plt.title('Temperature vs mass')
        plt.xlabel(r'$M/M_{sun}$')
        plt.ylabel(r'$T[MK]$')
        
        plt.show()
    
    def sanity_interpolation(self):
        """
        Sanity check for interpolation of the function made for the opacity table.
        """
        self.readfile('opacity.txt')
        
        log10_R = np.array((-6.00, -5.95, -5.80, -5.70, -5.55, -5.95, -5.95, -5.95,\
                            -5.80, -5.75, -5.70, -5.55, -5.50))
        log10_T = np.array((3.750, 3.755, 3.755, 3.755, 3.755, 3.770, 3.780, 3.795,\
                            3.770, 3.775, 3.780, 3.795, 3.800))
        
        print('  log10_T   log10_R   log10_K   K(SI)')
        print('---------------------------------------')
        for i in range(len(log10_T)):
            kappa = 10**(self.log_kappa(log10_R[i], log10_T[i]))
            kappa = float(kappa*(1e3/1e4))
            print('{:8.3f} {:9.2f} {:9.2f} {:9.2e}'.format(log10_T[i], log10_R[i],\
                                                      float(self.log_kappa(log10_R[i], log10_T[i])), kappa))
        
    def sanity_epsilon(self, T, rho):
        """
        Sanity test for epsilon*rho from appendix C.
        """
        r_Q = self.epsilon(T, rho)
    
        r_Q *= rho
        
        print('Energy production:')
        print('Temp = %.2e K' %T)
        print('-------------------------------------')
        for i in range(len(r_Q)):
            print('{:.2e}'.format(r_Q[i]))

# =============================================================================

# Sun's variables (for epsilon sanity check):

rho       = 1.62e5                      # kg/m^3
temp_core = 1.57e7                      # K
temp_2    = 1e8                         # K

# =============================================================================

if __name__ == '__main__':
    a = StellarCore()
    best_values = [a.R0*0.49195, a.T0*0.8025, a.rho0*0.7800]
    a.readfile('opacity.txt')
    #a.plot(a.M0, np.array([a.R0, a.T0, a.rho0]), test_R=True)
    a.plot(a.M0, np.array([a.R0, a.T0, a.rho0]), exp=True)
    #a.plot(a.M0, np.array(best_values))
    #a.plot(a.M0, np.array((a.R0, a.T0, a.rho0)), find=True)
    #a.finding_star(a.M0)
    #a.experiments(a.M0)
    #a.print_values(a.M0)
    
    #a.sanity_plot(a.M_sol)
    #a.sanity_interpolation()
    #a.sanity_epsilon(temp_2, rho)