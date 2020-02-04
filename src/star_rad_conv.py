import numpy as np
import matplotlib.pyplot as plt
import constants as c
from scipy.interpolate import interp2d

class Convection:
    """
    Variables that is to stay fixed at all times, and some constants.
    """
    L0 = c.L_sol                 # Solar luminosity; W
    M0 = c.M_sol                 # Solar mass; kg
    
    X    = 0.7                   # Hydrogen-1
    Y    = 0.29                  # Helium-3
    Y_3  = 1e-10                 # Helium-4
    Z    = 0.01                  # "Metals"
    Z_73 = 1e-13                 # Lithium-7
    Z_74 = 1e-13                 # Berrylium-7
    
    delta  = 1
    del_ad = 2/5
    
    def __init__(self, alpha=1, my=0.6182380216):
        """
        Initial parameters for the star.
        """
        self.alpha = alpha
        self.my    = my                         # kg
        self.R0    = c.R_sol                    # m
        self.rho0  = 1.42e-7*c.rho_sol          # kg m^-3
        self.T0    = c.T_sol_surf               # K
        self.c_P   = 5/2*c.k_B/(my*c.m_u)       # J kg^-1 K^-1
    
    def del_radiation(self, rho, T, M, R, kappa, L):
        """
        Calculating the radiative temperature gradient.
        """
        
        H_P = self.H_P(T, M, R)
        
        return (3*L*kappa*rho*H_P)/(64*np.pi*c.sigma*T**4*R**2)
    
    def del_star(self, rho, T, M, R, kappa, L, test=True):
        """
        Calculating the stellar temperature gradient.
        kwarg:  If test=True, we have instabilty in the mass shell, i.e., energy
                is transported through convection.
        """
                
        H_P = self.H_P(T, M, R)
        
        if test:
            F_C = self.F_C(rho, T, M, R, kappa, L)
        else:
            F_C = 0
        
        return (3*kappa*rho*H_P)/(16*c.sigma*T**4)*(L/(4*np.pi*R**2) - F_C)
    
    def del_parcel(self, rho, T, M, R, kappa, L):
        """
        Calculating the temperature gradient for the parcel.
        """
        
        del_s = self.del_star(rho, T, M, R, kappa, L)
        xi    = self.xi(rho, T, M, R, kappa, L)
        
        return del_s - xi**2
    
    def F_C(self, rho, T, M, R, kappa, L):
        """
        Calculating the convective flux.
        """
        
        c_P   = self.c_P
        delta = self.delta
        
        H_P = self.H_P(T, M, R)
        g   = self.gravity(M, R)
        l_m = self.alpha*H_P
        xi  = self.xi(rho, T, M, R, kappa, L)
        
        return rho*c_P*T*np.sqrt(g*delta)*H_P**(-3/2)*(l_m/2)**2*xi**3
    
    def F_R(self, rho, T, M, R, kappa, L):
        """
        Calculating the radiative flux.
        """
        
        del_s = self.del_star(rho, T, M, R, kappa, L)
        H_P   = self.H_P(T, M, R)
        
        return (16*c.sigma*T**4)/(3*kappa*rho*H_P)*del_s
    
    def gravity(self, M, R):
        """
        Calculating the gravitational acceleration for a given mass and radius,
        e.g., the mass and the average radius of the earth returns the value
        g = 9.82; m s^-2
        """
        
        return c.G*M/R**2
        
    def U(self, rho, T, M, R, kappa, H_P):
        """
        Calculating the variable "U" from exercise 5.12.
        """
        g = self.gravity(M, R)
        return (64*c.sigma*T**3)/(3*kappa*rho**2*self.c_P)*np.sqrt(H_P/(g*self.delta))
    
    def H_P(self, T, M, R):
        """
        Calculating the pressure scale height, which is defined as the height
        traveled which the pressure decreases by a factor of 1/e.
        """
        
        return c.k_B*T/(self.gravity(M, R)*self.my*c.m_u)
    
    def l_m(self, rho, T, M, R):
        """
        Mixing length. Generally assumed to be between 1/2 and 2 times the
        pressure scale height.
        """
        H_P = self.H_P(T, M, R)
        
        return self.alpha*H_P
    
    def xi(self, rho, T, M, R, kappa, L):
        """
        Function for finding xi-value (xi = (del_s - del_p)^(1/2)) from the 
        third degree polynomial, where xi's defined as (del_star - del_p)^(1/2).
        P is an array with the coefficients before xi^3, x^2, x^1 and xi^0
        respectively.
        """
        
        alpha = self.alpha
        
        H_P = self.H_P(T, M, R)
        U   = self.U(rho, T, M, R, kappa, H_P)
        l_m = alpha*H_P
        
        del_rad = self.del_radiation(rho, T, M, R, kappa, L)
        
        P = np.array([1, U/l_m**2, 4*U**2/(l_m)**4,
                      -(U/l_m**2)*(del_rad - self.del_ad)])
        xi      = np.roots(P)
        xi_real = xi[np.where(np.imag(xi) == 0)]
        
        if len(xi_real) > 1:
            print("We've gotten more than one xi; try new values.")
            exit()
            
        return np.real(xi_real)
    
    def velocity(self, rho, T, M, R, kappa, L):
        """
        Velocity of the parcel as it moves out towards the surface of the star.
        """
        
        del_s = self.del_star(rho, T, M, R, kappa, L)
        del_P = self.del_parcel(rho, T, M, R, kappa, L)
        
        g   = self.gravity(M, R)
        H_P = self.H_P(T, M, R)
        l_m = self.alpha*H_P
        
        return np.sqrt(g*self.delta*l_m**2/(4*H_P))*np.sqrt(del_s - del_P)

class StellarCore(Convection):
    """
    Class from project one with some new implimentations.
    Variables that is to stay fixed at all times, and some constants.
    """
        
    X    = 0.7                      # Hydrogen-1
    Y    = 0.29                     # Helium-3
    Y_3  = 1e-10                    # Helium-4
    Z    = 0.01                     # "Metals"
    Z_73 = 1e-13                    # Lithium-7
    Z_74 = 1e-13                    # Berrylium-7
    
    MeV_J = 1.60217662e-19*1e6      # Converting from MeV to J
    a     = 4*c.sigma/c.c             # J/m^3 K^4
    
    def __init__(self, alpha=1, my=0.6182380216, R0=c.R_sol, rho0=134*1.42e-7*1.408e3,
                 T0=5778, M0=c.M_sol):
        super().__init__(alpha, my)
        """
        Initial parameters for the star.
        """
        self.P0   = self.pressure(rho0, T0)
        self.M0   = M0                      # Mass [kg]
        self.call_readfile = False
    
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
        self.call_readfile = True
    
    def lambd_ik(self, T, rho):
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
                /(c.N_A*1e6)
        
        l_33  = (6.04e10*T_9**(-2/3)*np.exp(-12.276*T_9**(-1/3))\
                 *(1 + 0.034*T_9**(1/3) - 0.522*T_9**(2/3)\
                   - 0.124*T_9 + 0.353*T_9**(4/3) + 0.213*T_9**(-5/3)))\
                /(c.N_A*1e6)
        
        l_34  = (5.61e6*T_9_34**(5/6)*T_9**(-3/2)*np.exp(-12.826*T_9_34**(-1/3)))\
                /(c.N_A*1e6)
        
        l_7e  = (1.34e-10*T_9**(-1/2)*(1 - 0.537*T_9**(1/3) + 3.86*T_9**(2/3)\
                 + 0.0027*T_9**(-1)*np.exp(2.515e-3*T_9**(-1))))/(c.N_A*1e6)
        
        l_71_ = (1.096e9*T_9**(-2/3)*np.exp(-8.427*T_9**(-1/3))\
                 - 4.830e8*T_9_17**(5/6)*T_9**(-3/2)*np.exp(-8.472*T_9_17**(-1/3))\
                 + 1.06e10*T_9**(-3/2)*np.exp(-30.442*T_9**(-1)))\
                /(c.N_A*1e6)
        
        l_71  = (3.11e5*T_9**(-2/3)*np.exp(-10.262*T_9**(-1/3))\
                 + 2.53e3*T_9**(-3/2)*np.exp(-7.306*T_9**(-1)))\
                /(c.N_A*1e6)
        
        n_e = rho/(2*c.m_u)*(1 + self.X)
        if l_7e > 1.57e-7/n_e:
            l_7e = 1.57e-7/n_e
        
        lambdas = np.array([l_pp, l_33, l_34, l_7e, l_71_, l_71])
        
        return lambdas
    
    def epsilon(self, T, rho):
        """
            Calculating the total energy generation per unit mass.
            Starting by determining the reaction rate with respect to mass per
        step in the PP-chain, and then multiplying each step's reaction rate
        with its corresponding energy output.
            New implementation: Will also calculate the relative energy output
        per branch in the PP-chain, for a given temperature and density.
        """
        mu = c.m_u
        
        n_densities    = np.array((rho*self.X/mu, rho*self.Y_3/(3*mu),\
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
        
        lambdas = self.lambd_ik(T, rho)
        
        # Reaction rates per step in the PP-chain
        
        r     = np.zeros(6)
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
        
        # Calculating the relative energy output per branch in the PP-chain:
                      
        PPI    = (2*Q[0] + Q[1])*r[1]
        
        PPII   = (Q[0] + Q[2])*r[2]             # PPII, step 1
        PPII  += Q[3]*r[3]                      # PPII, step 2
        PPII  += Q[4]*r[4]                      # PPII, step 3
        
        PPIII  = (Q[0] + Q[5])*r[2]             # PPIII, step 1
        PPIII += Q[6]*r[5]                      # PPIII, step 2
        PPIII += Q[7]*r[5]                      # PPIII, step 3 (decay)
        PPIII += Q[8]*r[5]                      # PPIII, step 4 (decay)
        
        r_Q = r
    
        r_Q[0] *= Q[0]                          # W
        r_Q[1] *= Q[1]                          # W
        r_Q[2] *= Q[2]                          # W
        r_Q[3] *= Q[3]                          # W
        r_Q[4] *= Q[4]                          # W
        r_Q[5] *= (Q[6] + Q[7] + Q[8])          # W
        
        return r_Q, PPI, PPII, PPIII            # W
    
    def rho(self, P, T):
        return self.my*c.m_u*P/(c.k_B*T)    # kg/m^3
    
    def pressure(self, rho, T):
        return rho/(self.my*c.m_u)*c.k_B*T + self.a/3*T**4  # Pa

    def get_kappa(self, T, rho):
        """
        Function that takes T- and rho-values, and returns kappa in SI-units.
        """
        if not self.call_readfile:
            print("You haven't called for the function 'readfile()'!")
            exit()
        
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
    
    def solver(self, mass, initials, dm=1e-4, dynamic=True):
        """
        From project 1: Solving the four partial differential equations 
        respectively as they're given, using the Euler method.
        Now also testing if we have convection or not, and returning the 
        relative energy output per branch in the PP-chain, as well as del_rad,
        del_star, and the convective and radiative flux.
        """
        
        N = 10000
        p = 0.01
        
        R   = np.zeros(N)           # m
        P   = np.zeros(N)           # Pa
        L   = np.zeros(N)           # W
        T   = np.zeros(N)           # K
        rho = np.zeros(N)           # kg/m^3
        M   = np.zeros(N)           # kg
        eps = np.zeros((N, 3))      # Matrice for epsilon for each of PP-branches
        
        del_rad = np.zeros(N)       # Array for radiative gradient
        del_s   = np.zeros(N)       # Array for star gradient
        F_C     = np.zeros(N)       # Array for convective flux
        F_R     = np.zeros(N)       # Array for radiative flux
        
        L[0]   = self.L0
        M[0]   = mass
        R[0]   = initials[0]
        T[0]   = initials[1]
        rho[0] = initials[2]
        P[0]   = self.pressure(rho[0], T[0])
        
        # For calculating energy output for each PP-branch:
        
        _, PPI, PPII, PPIII = self.epsilon(T[0], rho[0])
        eps[0, 0] = PPI
        eps[0, 1] = PPII
        eps[0, 2] = PPIII
        
        kappa = self.get_kappa(T[0], rho[0])
        
        del_rad[0] = self.del_radiation(rho[0], T[0], M[0], R[0], kappa, L[0])
        test0      = del_rad[0] > self.del_ad
        del_s[0]   = self.del_star(rho[0], T[0], M[0], R[0], kappa, L[0], test0)
        
        dm = -abs(dm*self.M0)
        
        for i in range(N-1):
            test    = del_rad[i] > self.del_ad
            kappa   = self.get_kappa(T[i], rho[i])
            
            F_R[i] = self.F_R(rho[i], T[i], M[i], R[i], kappa, L[i])
            
            f_1 = 1/(4*np.pi*R[i]**2*rho[i])                        # dR/dM
            f_2 = -c.G*M[i]/(4*np.pi*R[i]**4)                       # dP/dM
            f_3 = np.sum(self.epsilon(T[i], rho[i])[0])             # dL/dM
            if test:
                """
                Runs if the convection criterion is met.
                """
                F_C[i] = self.F_C(rho[i], T[i], M[i], R[i], kappa, L[i])
                H_P    = self.H_P(T[i], M[i], R[i])
                
                f_4 = -self.del_star(rho[i], T[i], M[i], R[i], kappa, L[i], test)\
                            *T[i]/H_P*f_1                        # dT/dM
                
            else:
                f_4 = -(3*kappa*L[i])/(256*np.pi**2*c.sigma*R[i]**4*T[i]**3)  # dT/dM
            
            if dynamic:
                dm = [p*R[i]/f_1, p*P[i]/f_2, p*L[i]/f_3, p*T[i]/f_4]
                dm = -np.amin(np.abs(dm))
            
            R[i+1]   = R[i] + f_1*dm
            P[i+1]   = P[i] + f_2*dm
            L[i+1]   = L[i] + f_3*dm
            T[i+1]   = T[i] + f_4*dm
            rho[i+1] = self.rho(P[i+1], T[i+1])
            
            _, PPI, PPII, PPIII = self.epsilon(T[i+1], rho[i+1])
            eps[i+1, 0] = PPI
            eps[i+1, 1] = PPII
            eps[i+1, 2] = PPIII
            
            #print(rho[i+1], T[i+1], M[i], R[i+1], kappa, "\n")
            
            del_rad[i+1] = self.del_radiation(rho[i], T[i], M[i], R[i], kappa, L[i])
            del_s[i+1]   = self.del_star(rho[i], T[i], M[i], R[i], kappa, L[i], test)
            
            M[i+1]   = M[i] + dm
            
            if M[i+1] < 0:
                # Breaking the loop if the mass becomes negative
                
                R[i+1] = P[i+1] = L[i+1] = T[i+1] = rho[i+1] = P[i+1] = M[i+1]\
                       = del_rad[i+1] = del_s[i+1] = 0
                print('Mass < 0 at iteration {:d} of {:d}'\
                      .format(i, N))
                break
            
        return R[:i+1], P[:i+1], L[:i+1], T[:i+1], rho[:i+1], M[:i+1],\
               del_rad[:i+1], del_s[:i+1], F_C[:i+1], F_R[:i+1], eps[:i+1]
        
    def plot(self, mass, initials):
        """
        From project 1
        
        Plotting L(R), M(R), P(R), rho(R) and T(R) respectively where the R0-,
        T0- and rho0-values are given from the argument "initials".
        """
        
        R, P, L, T, rho, M, _, _, _, _, _ = self.solver(mass, initials)
        
        M0   = self.M0
        R0   = initials[0]
        T0   = initials[1]
        rho0 = initials[2]
        
        plt.subplot(3,2,1)
        plt.plot(R/R0, L/self.L0)
        plt.plot(np.ones(len(R))*R[(np.abs(L/c.L_sol - 0.995)).argmin()]/R0,
                 R/R0, "r--", label=r'$0.995L_{0}$')    # Plotting the core radius
        plt.title('L(R)', fontsize=19)
        plt.ylabel(r'$L/L_{0}$', fontsize=19)
        plt.xlabel(r"R/R_0", fontsize=19)
        plt.legend(loc='lower right', fontsize=17)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        ax1 = plt.subplot(3,2,2)
        plt.plot(R/R0, M/M0)
        plt.title('M(R)', fontsize=19)
        plt.ylabel(r'$M/M_{0}$', fontsize=19)
        plt.xlabel(r"R/R_0", fontsize=19)
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position('right')
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        plt.subplot(3,2,3)
        plt.semilogy(R/R0, P/self.P0)
        plt.title('P(R)', fontsize=19)
        plt.ylabel(r'$P/P_{0}$', fontsize=19)
        plt.xlabel(r"R/R_0", fontsize=19)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        ax2 = plt.subplot(3,2,4)
        plt.semilogy(R/R0, rho/rho0)
        plt.title(r'$\rho(R)$', fontsize=19)
        plt.ylabel(r'$\rho/\rho_{0}$', fontsize=19)
        plt.xlabel(r"R/R_0", fontsize=19)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        
        ax3 = plt.subplot(3,2,5)
        Label = r"$R_0 = {:.2f}R_\odot$" "\n" r"$T_0 = {:d}\ K$" "\n"\
                 r"$\rho_0 = {:.3e}\ g\ cm^{{-3}}$".format(R[0]/c.R_sol, int(T0), rho0)
        plt.plot(R/R0, T/1e6, label=Label)
        plt.title('T(R)', fontsize=19)
        plt.ylabel(r'$T [MK]$', fontsize=19)
        plt.xlabel(r"R/R_0", fontsize=19)
        chartBox = ax3.get_position()
        ax3.set_position([chartBox.x0, chartBox.y0, chartBox.width,\
                         chartBox.height])
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)

        ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 0.9),\
                   fontsize=17, shadow=True, ncol=2)
                
        plt.show()
        
def sanity_values():
    """
    Sanity test (example 5.1).
    """
    T, rho, R, M, L = 0.9e6, 55.9, 0.84*c.R_sol, 0.99*c.M_sol, c.L_sol
    kappa, alpha    = 3.98, 1
    a               = Convection(alpha, my=0.6)
    
    del_rad = a.del_radiation(rho, T, M, R, kappa, L)
    
    H_P = a.H_P(T, M, R)
    U_  = a.U(rho, T, M, R, kappa, H_P)
    xi  = a.xi(rho, T, M, R, kappa, L)
    v   = a.velocity(rho, T, M, R, kappa, L)
    F_C = a.F_C(rho, T, M, R, kappa, L)
    F_R = a.F_R(rho, T, M, R, kappa, L)
    
    del_s = a.del_star(rho, T, M, R, kappa, L)
    
    print("del_ad <   del_p  <   del_s  < del_rad")
    print("  {0:.1f}  < {1:.6f} < {2:.6f} < {3:.5f}".format(a.del_ad,
          a.del_parcel(rho, T, M, R, kappa, L)[0], del_s[0], del_rad))
    
    return del_rad, H_P/1e6, U_, xi, del_s, v, F_C/(F_C + F_R), F_R/(F_C + F_R)

def print_sanity():
    print("-------------------------")
    print("       Sanity test       ")
    print("-------------------------")
    
    x = np.zeros(8)
    x += sanity_values()
    
    print("")
    print("     del_rad = {:.2f}".format(x[0]))
    print("          HP = {:2.1f}".format(x[1]))
    print("           U = {:.2e}".format(x[2]))
    print("          xi = {:.3e}".format(x[3]))
    print("    del_star = {:.3f}".format(x[4]))
    print("           v = {:2.2f}".format(x[5]))
    print("  FC/(FC+FR) = {:.2f}".format(x[6]))
    print("  FR/(FC+FR) = {:.2f}".format(x[7]))

def sanity_dels():
    """
    1st sanity plot.
    """
    T0, rho0, R0, M0 = c.T_sol_surf, 1.42e-7*c.rho_sol, c.R_sol, c.M_sol
    alpha            = 1
    initials         = np.array([R0, T0, rho0])
    
    b = StellarCore(alpha=alpha, R0=R0, rho0=rho0, T0=T0, M0=M0)
    b.readfile('opacity.txt')
    
    R, _, _, _, _, _,del_rad, del_s, _, _, _ = b.solver(M0, initials)
    
    l = len(R)
    
    del_ad = np.ones(l)*b.del_ad
    
    plt.semilogy(R/c.R_sol, del_rad, label=r"$\nabla_{rad}$")
    plt.semilogy(R/c.R_sol, del_s, label=r"$\nabla^{*}$")
    plt.semilogy(R/c.R_sol, del_ad, label=r"$\nabla_{ad}$")
    plt.ylim(1e-1, 1e3)
    plt.xlim(0, 1.04)
    plt.legend(loc="best")
    
    plt.show()

def sanity_cross(best=False):
    """
    2nd sanity plot.
    """
    
    T0, rho0, R0, M0 = c.T_sol_surf, 1.42e-7*c.rho_sol, c.R_sol, c.M_sol
    alpha            = 1
    initials         = np.array([R0, T0, rho0])
    
    b = StellarCore(alpha=alpha, R0=R0, rho0=rho0, T0=T0, M0=M0)
    b.readfile('opacity.txt')
    
    R_values, _, L_values, _, _, _, _, _, F_C_list, _, _ = b.solver(M0, initials)
    
    R_values = R_values/c.R_sol
    L_values = L_values/c.L_sol
    
    n = len(R_values)
    
    plt.figure()
    fig = plt.gcf() # get current figure
    ax = plt.gca()  # get current axis
    
    rmax       = 1.2
    core_limit = 0.995
    
    ax.set_xlim(-rmax,rmax)
    ax.set_ylim(-rmax,rmax)
    ax.set_aspect('equal')	# make the plot circular
    
    if best:
        show_every = 50
    else:
        show_every = 5
        
    j = show_every
    
    for k in range(0, n-1):
    	j += 1
    	if j >= show_every:	# don't show every step - it slows things down
    		if(L_values[k] > core_limit):	# outside core
    			if(F_C_list[k] > 0.0):		# convection
    				circR = plt.Circle((0,0),R_values[k],color='red',fill=False)
    				ax.add_artist(circR)
    			else:				# radiation
    				circY = plt.Circle((0,0),R_values[k],color='yellow',fill=False)
    				ax.add_artist(circY)
    		else:				# inside core
    			if(F_C_list[k] > 0.0):		# convection
    				circB = plt.Circle((0,0),R_values[k],color='blue',fill = False)
    				ax.add_artist(circB)
    			else:				# radiation
    				circC = plt.Circle((0,0),R_values[k],color='cyan',fill = False)
    				ax.add_artist(circC)
    		j = 0
    circR = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='red',fill=True)		
                    # These are for the legend (drawn outside the main plot)
    circY = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='yellow',fill=True)
    circC = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='cyan',fill=True)
    circB = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='blue',fill=True)
    ax.legend([circR, circY, circC, circB],\
              ['Convection outside core', 'Radiation outside core',
               'Radiation inside core', 'Convection inside core'])
                # only add one (the last) circle of each colour to legend
    plt.xlabel(r'$r_{\odot}$')
    plt.ylabel(r'$r_{\odot}$')
    plt.title('Cross-section of star')
    
    # Show all plots
    plt.show()

def fluxes():
    """
    Plotting F_C(R) & F_R(R).
    """
    T0, rho0, R0, M0 = c.T_sol_surf, 134*1.42e-7*c.rho_sol, 1.251*c.R_sol, c.M_sol
    initials         = np.array([R0, T0, rho0])
    
    b = StellarCore()
    b.readfile("opacity.txt")
    R, _, _, _, _, _, _, _, F_C, F_R, _ = b.solver(M0, initials)
    
    F_tot = F_C + F_R
    
    plt.plot(R/R0, F_C/F_tot, label=r"$F_C$")
    plt.plot(R/R0, F_R/F_tot, label=r"$F_R$")
    plt.title("Normalized flux", fontsize=15)
    plt.xlabel(r"$R/R_0$", fontsize=15)
    plt.ylabel(r"$F_i/F_{\mathrm{tot}}$", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
        
    plt.legend(loc="center left", fontsize=15)
    
    plt.show()

def epsilons():
    """
    Plotting branches of PP-chains (as a function of R) for the "best" star.
    """
    
    T0, rho0, R0, M0 = c.T_sol_surf, 134*1.42e-7*c.rho_sol, 1.251*c.R_sol, c.M_sol
    initials         = np.array([R0, T0, rho0])
    
    b = StellarCore()
    b.readfile("opacity.txt")
    R, _, _, _, _, _, _, _, F_C, F_R, eps = b.solver(M0, initials)
    
    eps_tot = np.sum(eps, axis=1)
    
    plt.plot(R/R0, eps[:, 0]/eps_tot, label="PPI")
    plt.plot(R/R0, eps[:, 1]/eps_tot, label="PPII")
    plt.plot(R/R0, eps[:, 2]/eps_tot, label="PPIII")
    plt.title("Relative energy production from each of the branches of the PP-chain",
              fontsize=15)
    plt.xlabel(r"$R/R_0$", fontsize=15)
    plt.ylabel(r"$\epsilon/\epsilon_{tot}$", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    plt.legend(loc="best", fontsize=15)
    
    plt.show()

def plot_nablas(ax, T0=c.T_sol_surf, rho0=1.42e-7*c.rho_sol, R0=c.R_sol,
                col=1, temp=False, dens=False, radi=False):
    """
    Plotting nablas (as a function of R).
    """
    
    initials = np.array([R0, T0, rho0])
    
    b = StellarCore(alpha=1)
    b.readfile("opacity.txt")
    
    R, _, L, _, _, _, del_rad, del_s, _, _, _ = b.solver(c.M_sol, initials)
    
    del_ad = np.ones(len(R))*b.del_ad
        
    ax.set_xlabel(r"$R/R_0$", fontsize=15)
    ax.set_ylabel(r"$\nabla_i$", fontsize=15)

    ax.set_title(r"$R_0 = ${0:.2f}$R_\odot$; $\rho_0$ = {1:.0e} g cm$^{{-3}}$; $T_0$ = {2:d} K"
                 .format(R0/c.R_sol, rho0, int(T0)), fontsize=15)
        
    ax.set_ylim(1e-1, np.max(del_rad))
    ax.set_xlim(-0.1, 1.1)
    
    ax.semilogy(R/R0, del_rad, label=r"$\nabla^*$")                         # del_rad
    ax.semilogy(R/R0, del_s, label=r"$\nabla_{rad}$")                       # del_star
    ax.semilogy(R/R0, del_ad, label=r"$\nabla_{ad}$")                       # del_ad
    ax.plot(np.ones(len(R))*R[(np.abs(L/c.L_sol - 0.995)).argmin()]/R0,     # core radius
            np.linspace(0.2, 1.2, len(R)), label=r"Stellar core radius")
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    
    ax.legend(loc="best", ncol=col, fontsize=15)

def cross_section(ax, T0=c.T_sol_surf, rho0=1.42e-7*c.rho_sol, R0=1.4*c.R_sol, M0=c.M_sol,
                  get_plot=False, plotting=True):
    """
    Plotting the cross section of the star with some given parameters.
    """
        
    alpha            = 1
    initials         = np.array([R0, T0, rho0])
    
    b = StellarCore(alpha=alpha, R0=R0, rho0=rho0, T0=T0, M0=M0)
    b.readfile('opacity.txt')
    
    R, _, L, _, _, _, _, _, F_C, _, _ = b.solver(M0, initials)
    
    R = R/R0
    L = L/c.L_sol
    
    n = len(R)
    
    rmax       = 1.2
    core_limit = 0.995
    
    ax.set_xlim(-rmax,rmax)
    ax.set_ylim(-rmax,rmax)
    ax.set_aspect('equal')	# make the plot circular
    
    show_every = 5
    j = show_every        
    
    for k in range(0, n-1):
    	j += 1
    	if j >= show_every:	# don't show every step - it slows things down
    		if(L[k] > core_limit):	# outside core
    			if(F_C[k] > 0.0):		# convection
    				circR = plt.Circle((0,0),R[k],color='red',fill=False)
    				ax.add_artist(circR)
    			else:				# radiation
    				circY = plt.Circle((0,0),R[k],color='yellow',fill=False)
    				ax.add_artist(circY)
    		else:				# inside core
    			if(F_C[k] > 0.0):		# convection
    				circB = plt.Circle((0,0),R[k],color='blue',fill = False)
    				ax.add_artist(circB)
    			else:				# radiation
    				circC = plt.Circle((0,0),R[k],color='cyan',fill = False)
    				ax.add_artist(circC)
    		j = 0
    circR = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='red',fill=True)		
                    # These are for the legend (drawn outside the main plot)
    circY = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='yellow',fill=True)
    circC = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='cyan',fill=True)
    circB = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='blue',fill=True)
    ax.legend([circR, circY, circC, circB],\
              ['Convection outside core', 'Radiation outside core',
               'Radiation inside core', 'Convection inside core'], fontsize=15)
                # only add one (the last) circle of each colour to legend
    ax.set_xlabel(r"$R_0\ ( = {:.2f}R_\odot)$".format(R0/c.R_sol), fontsize=15)
    ax.set_ylabel(r"$R_0\ ( = {:.2f}R_\odot)$".format(R0/c.R_sol), fontsize=15)
    ax.set_title('Cross-section of the star', fontsize=15)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    
def testing(T0=c.T_sol_surf, rho0=1.42e-7*c.rho_sol, R0=c.R_sol,
            test_all=False):
    """
    Testing different parameters for our star.
    """
    
    l = 2
    
    if test_all:
        plt.subplots(3, 2, figsize=(16,9))
        
        T0_values = np.linspace(0.5, 2, l)*c.T_sol_surf
        rho0_values = np.linspace(0.5, 2, l)*1.42e-7*c.rho_sol
        R0_values = np.linspace(0.5, 2, l)*c.R_sol
    
        for i in range(3*l):
            ax = plt.subplot(3, 2, i+1)
            
            if i < 2:
                plot_nablas(ax, T0=T0_values[i], col=2, temp=True)
            
            elif 2 <= i <= 3:
                plot_nablas(ax, rho0=rho0_values[i - 2], col=2, dens=True)
                
            elif 4 <= i <= 5:
                plot_nablas(ax, R0=R0_values[i - 4], col=2, radi=True)
            
            if i%2 == 1:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position('right')
            
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
        
    else:
        """
        Since a varying temperature doesn't do much, we want to consider only
        varying radius and density.
        """
        plt.subplots(2, 2, figsize=(16,9))
        
        rho0_values = np.linspace(111.5, 164, l)*1.42e-7*c.rho_sol
        R0_values = np.linspace(1.2, 1.3, l)*c.R_sol
        
        for i in range(2*l):
            ax = plt.subplot(2, 2, i+1)
            
            if i < 2:
                plot_nablas(ax, rho0=rho0_values[i], R0=R0_values[-1 - i], dens=True)
                
            elif 2 <= i <= 3:
                plot_nablas(ax, rho0=rho0_values[i - 2], R0=R0_values[i - 2], radi=True)
            
            if i%2 == 1:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position('right')
            
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

    plt.tight_layout(pad=0.5)
    
    plt.show()

def final_plot(cross=False):
    """
    Plotting the 'best' star with the 'best' values found.
    """
    T0   = c.T_sol_surf
    rho0 = 134*1.42e-7*c.rho_sol
    R0   = 1.251*c.R_sol
    
    if cross:
        _, ax = plt.subplots(figsize=(9,9))  # get current axis
        cross_section(ax, T0=T0, rho0=rho0, R0=R0)
        
    else:
        _, ax = plt.subplots(figsize=(9,9))
        ax = plt.subplot(2, 1, 1)
        plot_nablas(ax, T0=T0, rho0=rho0, R0=R0)
        
        ax = plt.subplot(2, 1, 2)
        plot_nablas(ax, T0=T0, rho0=rho0, R0=R0)
    
    plt.tight_layout(pad=0.5)
    plt.show()
    
if __name__ == '__main__':
    print_sanity()
    sanity_dels()
    sanity_cross()
    
    #fluxes()
    #epsilons()
    #final_plot(cross=True)
    
    #testing(test_all=False)
    
    #b = StellarCore(alpha=1)
    #b.readfile("opacity.txt")
    #b.plot(b.M0, np.array([1.251*c.R_sol, c.T_sol_surf, 134*1.42e-7*c.rho_sol]))