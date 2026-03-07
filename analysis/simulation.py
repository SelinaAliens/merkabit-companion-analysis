import numpy as np
import numba

class FHTSimulation:

    """ Class to simulate pure-state DTC circuits
    (uses quantum trajectories for noise). """

    def __init__(self, L, J0 = np.pi/4., dJ = np.pi/8., relative_2g = 0.97, h0 = 0., dh = 2*np.pi):
        self.L = L
        self.dim = dim = 2**L
        self.g = np.pi/2. * relative_2g
        #basis & operators
        ind = np.arange(dim)
        basis = np.zeros([dim, L], dtype = np.int)
        for i in range(L):
            basis[:,i] = np.mod(ind,2)
            ind //= 2
        self.basis = basis[:,::-1]
        self._lookup = {tuple(ket):i for i,ket in enumerate(self.basis)}
        self.lookup = lambda ket: self._lookup[tuple(ket)]
        self.Z = 1-2*self.basis
        self.ZZ = self.Z[:, 1:] * self.Z[:,:-1] # OBCs
        self.reset_Uz(J0, dJ, h0, dh)
        Hx = self.g*self.Z.sum(axis=1)
        self.Ux = np.exp(-1j*Hx)
        self.reset_psi() # default to all up

    def reset_psi(self, psi0=None):
        self.psi = np.zeros(self.dim, dtype=np.complex)
        if psi0 is None: # default to all-up
            self.psi[0] = 1.
        else:
            j = self.lookup(psi0)
            self.psi[j] = 1.

    def reset_Uz(self, J0 = 0.1, dJ = 0., h0 = 0., dh = 2*np.pi):
        L = self.L
        self.h = h0 + (2*np.random.rand(L)-1.)*dh
        self.J = J0 + (2*np.random.rand(L-1)-1)*dJ
        Hz_1 = np.dot(self.ZZ[:,::2], self.J[::2])
        Hz_2 = np.dot(self.ZZ[:,1::2], self.J[1::2]) + np.dot(self.Z, self.h)
        self.Uz_1 = np.exp(-1j*Hz_1)
        self.Uz_2 = np.exp(-1j*Hz_2)

    def noise(self, p, i0 = 0, i1 = None):
        if i1 is None:
            i1 = self.L
        for i in range(i0, i1): # sites where noise is acting
            rando = np.random.rand()
            if rando<p: # error occurs
                aux = self.psi.reshape(2**i, 2, -1)
                pauli = 'xyz'[int(3*rando/p)]
                if pauli=='x':
                    aux = aux[:,::-1]
                elif pauli=='z':
                    aux[:,1] = -aux[:,1]
                elif pauli=='y':
                    aux = aux[:,::-1]/1j # x/i
                    aux[:,1] = -aux[:,1] # z
                self.psi = aux.ravel()

    def change_basis(self):
        self.fwht(self.psi)

    @staticmethod
    @numba.jit()
    def fwht(a):
        """In-place Fast Walsh–Hadamard Transform of array a"""
        h = 1
        dim = len(a)
        while h < len(a):
            for i in range(0, len(a), h * 2):
                for j in range(i, i + h):
                    x = a[j]
                    y = a[j + h]
                    a[j] = x + y
                    a[j + h] = x - y
            h *= 2
        for i in range(len(a)): # normalization
            a[i] = a[i]/np.sqrt(len(a))

    def evolve(self, p = 1e-2):
        if p:
            # even bonds
            self.psi = self.Uz_1*self.psi
            self.noise(p)
            # odd bonds
            self.psi = self.Uz_2*self.psi
            self.noise(p, i0=1, i1=self.L-1)
        else: # all bonds
            self.psi = (self.Uz_1 * self.Uz_2) * self.psi
        self.change_basis()
        self.psi = self.Ux*self.psi
        if p: # 1-qubit noise on all sites
            self.noise(p/10.)
        self.change_basis()

    def Z_EV(self):
        return np.dot(np.abs(self.psi)**2, self.Z)

    def run(self, T, p=1e-2):
        """ Returns <Z_i> expectation values,
        related to autocorrelators by A(t) = Z_i(0) <Z_i(t)>
        (starting from Z product state) """
        out = []
        ts = np.arange(T)
        for t in ts:
            zs = self.Z_EV()
            out.append(zs)
            self.evolve(p=p)
        return ts, np.array(out)


class RandomStateSimulation(FHTSimulation):

    """ Subclass that implements scrambling circuits
    before the DTC circuit (for quantum typicality measurement). """

    def __init__(self, L, J0 = np.pi/4., dJ = np.pi/8., relative_2g = 0.97, h0 = 0., dh = 2*np.pi):
        super().__init__(L, J0=J0, dJ=dJ, relative_2g=relative_2g, dh=dh, h0=h0)
        self.CZ_e = np.exp(-1j*np.pi/4.*self.ZZ[:,::2].sum(axis=1))
        self.CZ_o = np.exp(-1j*np.pi/4.*self.ZZ[:,1::2].sum(axis=1))

    def get_random_Z_layer(self):
        h = np.random.rand(self.L)*2*np.pi
        Uz = np.exp(-1j*np.dot(self.Z, h))
        return Uz

    def state_prep_layer_1q(self):
        g = (0.4 + 0.2*np.random.rand())
        Ux = np.exp(-1j*np.pi/2.*g*self.Z.sum(axis=1))
        Uz = self.get_random_Z_layer()
        self.psi = Uz * self.psi
        self.change_basis()
        self.psi = Ux * self.psi
        self.change_basis()
        self.psi = Uz.conj() * self.psi

    def state_prep_layer(self):
        self.psi = self.CZ_e * self.psi
        self.state_prep_layer_1q()
        self.psi = self.CZ_o * self.psi
        self.state_prep_layer_1q()

    def state_prep(self, K, seed=None):
        if seed:
            np.random.seed(seed)
        for _ in range(K):
            self.state_prep_layer()

class RhoSimulation:

    """ Exact density matrix simulations.
    Maps density matrix to wavefunction in doubled Hilbert space.
    Implements noise as exact quantum channel. """

    def __init__(self, L, J0 = np.pi/4., dJ = np.pi/8., g = np.pi/2.*0.97, h0 = 0., dh = 2*np.pi):
        self.L = L
        self.dim = dim = 2**L
        # params
        self.g = g
        #basis & operators
        ind = np.arange(dim**2)
        basis = np.zeros([dim**2, 2*L], dtype = np.int)
        for i in range(2*L):
            basis[:,i] = np.mod(ind,2)
            ind //= 2
        self.basis = basis[:,::-1]
        self._lookup = {tuple(ket):i for i,ket in enumerate(self.basis)}
        self.lookup = lambda ket: self._lookup[tuple(ket)]
        Z = 1-2*self.basis
        self.ZL = Z[:,:L]
        self.ZR = Z[:,L:]
        self.ZZL = self.ZL[:,1:] * self.ZL[:,:-1]
        self.ZZR = self.ZR[:,1:] * self.ZR[:,:-1]
        self.reset_Uz(h0=h0,dh=dh,J0=J0,dJ=dJ) # defines self.h, self.Uz
        Hx = self.g * (self.ZL.sum(axis=1) - self.ZR.sum(axis=1))
        self.Ux = np.exp(-1j*Hx)
        self.reset_psi() # default to all up

    def reset_psi(self, psi0=None):
        self.psi = np.zeros(self.dim**2, dtype=np.complex)
        if psi0 is None: # default to all-up
            self.psi[0] = 1.
        else:
            j = self.lookup(psi0)
            self.psi[j] = 1.

    def reset_Uz(self, J0=np.pi/4., dJ=np.pi/8., h0=0., dh = 2*np.pi):
        self.h = h0 + (2*np.random.rand(L)-1)*dh
        self.J = J0 + (2*np.random.rand(L-1)-1)*dJ
        Hz_e = ( np.dot(self.ZZL[:,::2], self.J[::2])
                 - np.dot(self.ZZR[:,::2], self.J[::2])
                 + np.dot(self.ZL, self.h) - np.dot(self.ZR, self.h) )
        Hz_o = ( np.dot(self.ZZL[:,1::2], self.J[1::2])
                 - np.dot(self.ZZR[:,1::2], self.J[1::2]) )
        self.Uz_e = np.exp(-1j*Hz_e)
        self.Uz_o = np.exp(-1j*Hz_o)

    def noise(self, p=0.01):
        psi_tmp = self.psi
        for i in range(self.L):
            dim_A = 2**i
            dim_B = 2**(self.L-i-1)
            psi_tmp = psi_tmp.reshape((dim_A, 2, dim_B, dim_A, 2, dim_B))
            depolarized = np.einsum("ij,akbckd->aibcjd", np.eye(2)/2.,
                                    psi_tmp, optimize=True)
            psi_tmp = (1-p)*psi_tmp + p*depolarized
        self.psi = psi_tmp.ravel()

    @staticmethod
    @numba.jit()
    def fwht(a):
        """In-place Fast Walsh–Hadamard Transform of array a"""
        h = 1
        dim = len(a)
        while h < len(a):
            for i in range(0, len(a), h * 2):
                for j in range(i, i + h):
                    x = a[j]
                    y = a[j + h]
                    a[j] = x + y
                    a[j + h] = x - y
            h *= 2
        for i in range(len(a)): # normalization
            a[i] = a[i]/np.sqrt(len(a))

    def change_basis(self):
        self.fwht(self.psi)

    def evolve(self, p=0.01):
        self.psi = self.Uz_e*self.psi
        # two-qubit gates
        if p:
            self.noise(p)
        self.psi = self.Uz_o*self.psi
        if p:
            self.noise(p)
        # one-qubit gates
        self.change_basis()
        self.psi = self.Ux*self.psi
        if p:
            self.noise(p/10.)
        self.change_basis()

    def run(self, T, p):
        """ Returns expectation values of <Z_i Z_j>,
        used to compute chi^SG. """
        out = []
        ts = np.arange(T)
        for t in ts:
            z_aux = self.ZR[:self.dim,:].reshape(self.dim,self.L)
            zzs = np.einsum("jj,jx,jy->xy", self.psi.reshape(self.dim,-1),
                            z_aux, z_aux, optimize=True)
            out.append(zzs)
            self.evolve(p)
        return np.array(out)