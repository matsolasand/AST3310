import numpy as np
import matplotlib.pyplot as plt
import constants as c
from project_1 import StellarCore

class Convection:
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
    
    def __init__(self, alpha, my=0.6182380216):
        self.alpha = alpha
        self.my    = my
        self.R0    = c.R_sol
        self.rho0  = 1.42e-7*c.rho_sol
        self.T0    = c.T_sol_surf
        self.c_P   = 5/2*c.k_B/(my*c.m_u)
    
    def del_radiation(self, rho, T, M, R, kappa):
        """
        Returning del radiation
        """
        
        H_P = self.H_P(rho, T, M, R)
        
        return (3*self.L0*kappa*rho*H_P)/(64*np.pi*c.sigma*T**4*R**2)
    
    def del_star(self, rho, T, M, R, kappa):
        """
        Returning del star
        """
        
        alpha = self.alpha
        
        xi  = self.xi(rho, T, M, R, kappa)
        H_P = self.H_P(rho, T, M, R)
        U   = self.U(rho, T, M, R, kappa, H_P)
        l_m = alpha*H_P
        
        return xi**2 + 4/l_m*U/(alpha*H_P)*xi + self.del_ad
    
    def del_parcel(self, rho, T, M, R, kappa):
        """
        Returning del parcel
        """
        
        del_s = self.del_star(rho, T, M, R, kappa)
        xi    = self.xi(rho, T, M, R, kappa)
        
        return del_s - xi**2
    
    def F_C(self, rho, T, M, R, kappa):
        """
        Returning F_C
        """
        
        c_P   = self.c_P
        delta = self.delta
        
        del_s = self.del_star(rho, T, M, R, kappa)
        del_p = self.del_parcel(rho, T, M, R, kappa)
        
        H_P = self.H_P(rho, T, M, R)
        g   = self.gravity(M, R)
        l_m = self.alpha*H_P
        
        return rho*c_P*T*np.sqrt(g*delta)*H_P**(-3/2)*(l_m/2)**2*(del_s - del_p)**(3/2)
    
    def F_R(self, rho, T, M, R, kappa):
        """
        Return F_R---WOEFMEOM
        """
        
        del_s = self.del_star(rho, T, M, R, kappa)
        H_P   = self.H_P(rho, T, M, R)
        
        return (16*c.sigma*T**4)/(3*kappa*rho*H_P)*del_s
    
    def gravity(self, M, R):
        """
        Gravity.
        """
        return c.G*M/R**2
        
    def U(self, rho, T, M, R, kappa, H_P):
        """
        Fill in
        """
        g = self.gravity(M, R)
        return (64*c.sigma*T**3)/(3*kappa*rho**2*self.c_P)*np.sqrt(H_P/(g*self.delta))
    
    def H_P(self, rho, T, M, R):
        """
        Scale height
        """
        
        return c.k_B*T/(self.gravity(M, R)*self.my*c.m_u)
    
    def l_m(self, rho, T, M, R):
        """
        Mixing length
        """
        H_P = self.H_P(rho, T, M, R)
        
        return self.alpha*H_P
    
    def xi(self, rho, T, M, R, kappa):
        """
        Function for finding xi-value.
        """
        
        alpha = self.alpha
        
        H_P = self.H_P(rho, T, M, R)
        U   = self.U(rho, T, M, R, kappa, H_P)
        l_m = alpha*H_P
        
        del_rad = self.del_radiation(rho, T, M, R, kappa)
        
        p = np.array([1, U/(alpha*H_P)**2, (4/l_m)*(U**2/(alpha*H_P)**3),
                      -(U/(alpha*H_P)**2)*(del_rad - self.del_ad)])
        xi      = np.roots(p)
        xi_real = xi[np.where(np.imag(xi) == 0)]
        
        if len(xi_real) > 1:
            print("We've gotten more than one xi; try new values.")
            exit()
            
        return np.real(xi_real)
    
    def velocity(self, rho, T, M, R, kappa):
        """
        Velocity of parcel???
        """
        
        del_s = self.del_star(rho, T, M, R, kappa)
        del_P = self.del_parcel(rho, T, M, R, kappa)
        
        g   = self.gravity(M, R)
        H_P = self.H_P(rho, T, M, R)
        l_m = self.alpha*H_P
        
        return np.sqrt(g*self.delta*l_m**2/(4*H_P))*np.sqrt(del_s - del_P)
    
def sanity_values():
    """
    Sanity test
    """
    T, rho, R, M = 0.9e6, 55.9, 0.84*c.R_sol, 0.99*c.M_sol
    kappa, alpha = 3.98, 1
    a                = Convection(alpha, my=0.6)
    
    del_rad = a.del_radiation(rho, T, M, R, kappa)
    
    H_P = a.H_P(rho, T, M, R)
    l_m = alpha*H_P
    r_p = l_m/2
    U_  = a.U(rho, T, M, R, kappa, H_P)
    xi  = a.xi(rho, T, M, R, kappa)
    v   = a.velocity(rho, T, M, R, kappa)
    F_C = a.F_C(rho, T, M, R, kappa)
    F_R = a.F_R(rho, T, M, R, kappa)
    
    del_s = xi**2 + 2/r_p*U_/(alpha*H_P)*xi + a.del_ad
    
    print("del_ad <   del_p  <   del_s  < del_rad")
    print("  {0:.1f}  < {1:.6f} < {2:.6f} < {3:.5f}".format(a.del_ad,
          a.del_parcel(rho, T, M, R, kappa)[0], del_s[0], del_rad))
    
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

def sanity_del():
    """
    1st sanity plot
    """
    T, rho, R, M = 0.9e6, 55.9, 0.84*c.R_sol, 0.99*c.M_sol
    kappa, alpha = 3.98, 1
    
    a = Convection(alpha, my=0.6)
    b = StellarCore(R0=R, rho0=rho, T0=T, M0=M)
    
    R = np.linspace(c.R_sol, 1e-7, 1001)
    l = len(R)
    
    del_rad = np.zeros(l)
    del_s   = np.zeros(l)
    del_ad  = a.del_ad*np.ones(l)
    
    for i in range(l):
        """
        UPDATE rho, T, M EVERY TEIM!!!!
        """
        del_rad[i] = a.del_radiation(rho, T, M, R[i], kappa)
        del_s[i]   = a.del_star(rho, T, M, R[i], kappa)
    
    plt.semilogy(R/c.R_sol, del_rad)
    plt.semilogy(R/c.R_sol, del_s)
    plt.semilogy(R/c.R_sol, del_ad)
    
    plt.show()

if __name__ == "__main__":
    #a = Convection(1)
    print_sanity()
    sanity_del()