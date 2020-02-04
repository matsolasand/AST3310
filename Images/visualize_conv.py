import FVis3 as FVis
import numpy as np
import matplotlib.pyplot as plt
import constants as c

class Convection2D:

    def __init__(self):
        """
        Defining constants.
        """

        self.ny    = 100
        self.nx    = 300

        self.y     = np.linspace(0, 4e6, self.ny)
        self.dy    = self.y[1] - self.y[0]
        self.y0    = self.y[-1]

        self.x     = np.linspace(0, 12e6, self.nx)
        self.dx    = self.x[1] - self.x[0]
        self.x0    = self.x[-1]

        self.p     = 0.1

        self.g     = c.G*c.M_sol/c.R_sol**2
        self.mu    = 0.61
        self.nabla = 2/5 + 1e-4

        d_o_f      = 3
        self.gamma = (d_o_f + 2)/d_o_f

    def initialize(self, T0=5778, P0=1.8e8, time=200, perturbation=None):
        """
        initializing temperature, pressure, density and internal energy, as
        well as turning perturbation on or off.
        """

        if not (perturbation == "on" or perturbation == "off"):
            print("Choose parameter perturbation='on' or perturbation='off'")
            exit()

        self.called  = True   # Updating if initializer has been called

        mu, g, gamma, nabla = self.mu, self.g, self.gamma, self.nabla
        y, y0               = self.y, self.y0

        nx, ny   = self.nx, self.ny

        T_init   = np.zeros((ny, nx))
        P_init   = np.zeros((ny, nx))
        e_init   = np.zeros((ny, nx))
        rho_init = np.zeros((ny, nx))

        for i in range(nx):
            T_init[:, i] = T0 - mu*c.m_u*nabla*g/c.k_B*(y - y0)
            P_init[:, i] = (1
                            - mu*c.m_u*g*nabla*(y - y0)/(c.k_B*T0))**(1/nabla)\
                           *P0

        if perturbation == "on":
            T_init += 0.2*T_init*self.perturbation()

        e_init   = P_init/(gamma - 1)
        rho_init = mu*c.m_u/(c.k_B*T_init)*(gamma - 1)*e_init

        self.e, self.rho = e_init, rho_init
        self.u           = np.zeros((ny, nx))
        self.w           = np.zeros((ny, nx))
        self.T, self.P   = T_init, P_init

        self.time = time

    def timestep(self, time):
        """
        Calculating timestep.
        """
        u, w, rho, e = self.u[:], self.w[:], self.rho[:], self.e[:]

        rel_rho = np.max(np.abs(self.drho_dt/rho))
        rel_e   = np.max(np.abs(self.de_dt/e))
        rel_x   = np.max(np.abs(u/self.dx))
        rel_y   = np.max(np.abs(w/self.dy))

        delta_list = [rel_rho, rel_e, rel_x, rel_y]

        delta      = np.max(delta_list)

        self.dt    = self.p/delta

        if self.dt > 0.1:
            self.dt = 0.1

        if self.dt <= 1e-4:
            self.dt = 1e-4

    def boundary_conditions(self):
        """
        Boundary conditions for energy, density and velocity
        """
        g, mu, gamma = self.g, self.mu, self.gamma

        self.w[0, :]  = 0
        self.w[-1, :] = 0

        #self.u[0, :]  = self.u[1, :]
        #self.u[-1, :] = self.u[-2, :]

        self.u[0, :]  = 4/3*self.u[1, :] - 1/3*self.u[2, :]
        self.u[-1, :] = 4/3*self.u[-2, :] - 1/3*self.u[-3, :]

        self.e[0, :]  = (4*self.e[1, :] - self.e[2, :])\
                        /(3 - g*2*self.dy*mu*c.m_u/(c.k_B*self.T[0, :]))
        self.e[-1, :] = (4*self.e[-2, :] - self.e[-3, :])\
                        /(3 + g*2*self.dy*mu*c.m_u/(c.k_B*self.T[-1, :]))

        T = self.T[:]

        self.rho[0, :]  = (gamma - 1)*self.e[0, :]*mu*c.m_u/(c.k_B*T[0, :])
        self.rho[-1, :] = (gamma - 1)*self.e[-1, :]*mu*c.m_u/(c.k_B*T[-1, :])

    def central_x(self, func):
        """
        Central difference scheme in x-direction.
        """
        neg = np.roll(func, 1, axis=1)
        pos = np.roll(func, -1, axis=1)

        diff = (pos - neg)/(2*self.dx)

        return diff

    def central_y(self, func):
        """
        Central difference scheme in y-direction.
        """
        neg = np.roll(func, 1, axis=0)
        pos = np.roll(func, -1, axis=0)

        diff = (pos - neg)/(2*self.dy)

        return diff

    def upwind_x(self, func, v):
        """
        Upwind difference scheme in x-direction.
        """
        phi = np.zeros((self.ny, self.nx))

        neg = np.roll(func, 1, axis=1)
        pos = np.roll(func, -1, axis=1)

        phi[np.where(v >= 0)] = func[np.where(v >= 0)] - neg[np.where(v >= 0)]
        phi[np.where(v < 0)] = pos[np.where(v < 0)] - func[np.where(v < 0)]

        phi /= self.dx

        return phi

    def upwind_y(self, func, v):
        """
        Upwind difference scheme in y-direction.
        """
        phi = np.zeros((self.ny, self.nx))

        neg = np.roll(func, 1, axis=0)
        pos = np.roll(func, -1, axis=0)

        phi[np.where(v >= 0)] = func[np.where(v >= 0)] - neg[np.where(v >= 0)]
        phi[np.where(v < 0)] = pos[np.where(v < 0)] - func[np.where(v < 0)]

        phi /= self.dy

        return phi

    def hydro_solver(self):
        """
        Hydrodynamic equations solver.
        """
        if not self.called:
            print("Initialize not called!")
            exit()

        u   = np.nan_to_num(np.copy(self.u))
        w   = np.nan_to_num(np.copy(self.w))
        P   = np.nan_to_num(np.copy(self.P))
        e   = np.nan_to_num(np.copy(self.e))
        rho = np.nan_to_num(np.copy(self.rho))

        # d/dt for rho

        self.drho_dt  = - rho*(self.central_x(u) + self.central_y(w))\
                        - u*self.upwind_x(rho, u)\
                        - w*self.upwind_y(rho, w)

        # d/dt for rho*u

        self.drhou_dt = - rho*u*(self.upwind_x(u, u) + self.upwind_y(w, u))\
                        - u*self.upwind_x(rho*u, u)\
                        - w*self.upwind_y(rho*u, w)\
                        - self.central_x(P)

        # d/dt for rho*w

        self.drhow_dt = - rho*w*(self.upwind_y(w, w) + self.upwind_x(u, w))\
                        - w*self.upwind_y(rho*w, w)\
                        - u*self.upwind_x(rho*w, u)\
                        - self.central_y(P)\
                        - rho*self.g

        # d/dt for e

        self.de_dt    = - self.central_x(e*u) - self.central_y(e*w)\
                        - P*(self.central_x(u) + self.central_y(w))

        self.timestep(time=self.time)      # Calling timestep to define self.dt

        dt = self.dt
        de = self.de_dt*dt

        self.rho[:] = rho + self.drho_dt*dt
        self.e[:]   = e + de
        self.u[:]   = (rho*u + self.drhou_dt*dt)/self.rho
        self.w[:]   = (rho*w + self.drhow_dt*dt)/self.rho

        self.boundary_conditions()

        self.P[:] = (self.gamma - 1)*self.e[:]
        self.T[:] = self.P*self.mu*c.m_u/(c.k_B*self.rho)

        return dt

    def perturbation(self, show_pert=False):
        """
        Defining the gaussian perturbation if it is turned on.
        """
        mu_x  = self.x0/2
        mu_y  = self.y0/2

        sigma_y = 1e6
        sigma_x = sigma_y

        X, Y = np.meshgrid(self.x, self.y)

        gauss = np.exp(-((X - mu_x)**2/(2*sigma_x**2)
                         + (Y - mu_y)**2/(2*sigma_y**2)))

        if show_pert:
            """
            Showing the perturbation as it is
            """
            plt.contourf(X, Y, gauss)
            plt.colorbar()
            plt.show()

        return gauss

if __name__ == '__main__':
    vis = FVis.FluidVisualiser()

    bools    = np.array([True, False])
    see_pert = bools[1]
    sanity1  = bools[1]
    save_sim = bools[1]
    sanity2  = bools[1]

    if see_pert:
        """
        Showing the perturbation as it is
        """
        time  = 1

        q = Convection2D()
        q.initialize(time=time, perturbation="on")
        q.perturbation(show_pert=True)

    elif sanity1:
        """
        Sanity test
        """
        time = 60

        q = Convection2D()
        q.initialize(time=time, perturbation="off")

        vis.save_data(time, q.hydro_solver, rho=q.rho, u=q.u, w=q.w, e=q.e,
                      P=q.P, T=q.T, folder="Sanity data")

        vis.animate_2D("T", cmap = "plasma", extent = [0, 12, 0, 4],\
                       units = {"Lx": "Mm", "Lz": "Mm"}, quiverscale = 2,
                       video_name="Sanity test", folder="Sanity data",
                       save=bools[1])

        vis.delete_current_data()

    elif save_sim:
        """
        This is for simulating convection (with perturbation "on")
        """
        snaps = [0, 49, 99, 149, 199, 249]

        time  = 250

        q = Convection2D()
        q.initialize(time=time, perturbation="on")
        #q.perturbation(show_pert=True)

        vis.save_data(time, q.hydro_solver, rho=q.rho, u=q.u, w=q.w, e=q.e,
                      P=q.P, T=q.T, folder="Convection data")

        vis.animate_2D("T", cmap="plasma", extent=[0, 12, 0, 4],
                       units={"Lx": "Mm", "Lz": "Mm"}, quiverscale=2,
                       video_name="Convection", folder="Convection data",
                       save=bools[1])#, snapshots=snaps)

        #vis.animate_energyflux(folder="Convection data", extent=[0, 12, 0, 4],
        #                       video_name="Energy flux",
        #                       save=bools[1])#, snapshots=snaps)

        vis.delete_current_data()

    else:
        """
        This is made for simulating already saved values.
        """

        if sanity2:
            vis.animate_2D("T", cmap="plasma", extent=[0, 12, 0, 4],
                           units={"Lx": "Mm", "Lz": "Mm"}, quiverscale=2,
                           video_name="Sanity test", folder="Sanity data",
                           save=bools[1])#, snapshots=snaps)
        else:
            snaps = []

            for i in range(250):
                if i%20 == 0:
                    snaps.append(i)

            vis.animate_2D("T", cmap="plasma", extent=[0, 12, 0, 4],
                           units={"Lx": "Mm", "Lz": "Mm"}, quiverscale=2,
                           video_name="Convection", folder="Convection data",
                           save=bools[1])#, snapshots=snaps)

            #vis.animate_energyflux(folder="Convection data", extent=[0, 12, 0, 4],
            #                       video_name="Energy flux",
            #                       save=bools[1])#, snapshots=snaps)