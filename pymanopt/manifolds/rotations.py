from __future__ import division

import numpy as np
import numpy.linalg as la
import numpy.random as rnd
from scipy.misc import comb

from pymanopt.manifolds.manifold import Manifold
from pymanopt.tools.multi import multisym, multiprod, multitransp, multiskew

def randskew(n, N=1):
    # TODO: figure out how to do find like in matlab
    # I, J = np.find()

    t = np.triu(np.ones(n), 1)
    I = t[t > 0]

    # TODO: check this against manopt version
    K = np.tile(np.arange(0:N), (n*(n-1)/2, 1))

    raise NotImplementedError
class Rotation(Manifold):
    """
    Special orthogonal group is the manifold of orthogonal matrices R of size 
    n x n with determinant 1. Useful for vision and state estimation problems
    involving rotations in 2D or 3D space.

    Examples:
    Create a manifold of 3D rotation matrices:
    manifold = Rotation(3)
    """

    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        self._dim = k*comb(n, 2)
        self._typicaldist = np.pi*np.sqrt(n*k)

        if self._k == 1:
            self._name = "Rotations manifold SO({})".format(self._n)
        else:
            self._name = "Product rotations manifold SO({})^{} matrices".format(
                         self._n, self._k)

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return self._dim

    @property
    def typicaldist(self):
        return self._typicaldist

    def log(self, x, y):
        # From manopt
        u = multiprod(multitransp(x), y)
        for i in range(0, k):
            u[:,:,i] = np.real(la.logm(u[:,:,i]))
        return multiskew(u)

    def dist(self, x, y):
        # From manopt
        return la.norm(x, self.log(x,y))

    def inner(self, x, u, v):
        # TODO: figure out if tensordot is needed and if multitranspose is needed
        # Need to understand why x isn't used
        return np.tensordot(u.T, v)

    def proj(self, X, G):
        return multiskew(multiprod(multitranspose(X), G))

    egrad2rgrad = proj

    def ehess2rhess(self, x, egrad, ehess, u):
        xt = multitranspose(x)
        xt_egrad = multiprod(xt, egrad)
        sym_xt_egrad = multisym(xt_egrad)
        xt_ehess = multiprod(xt, ehess)
        return multiskew(xt_ehess - multiprod(u, sym_xt_egrad))

    def retr(self, X, G, t=1):
        tG = t*G
        Y = X + multiprod(X, tG)
        for i in range(0, k):
            Q, R = la.qr(Y[:,:,i])
            # TODO: figure out if this is needed (from manopt), it 
            # has to do with old matlab versions
            Y[:,:,i] = Q*np.diag(np.sign(np.sign(np.diag(R)) + 0.5))
        return Y

    def norm(self, x, u):
        # TODO: figure out if slice is needed like in manopt
        return la.norm(u[:])

    def rand(self):
        if n == 1:
            return np.ones(1, 1, k)
        R = np.zeros((n,n,k))
        for i in range(0,k):
            A = rnd.randn(n, n)
            Q, RR = la.qr(A)
            Q = np.dot(Q, np.diag(np.sign(np.diag(RR))))

            if la.det(Q) < 0
                Q[:, [0,1]] = Q[:, [1,0]]
            R[:,:,i] = Q
        return R

    def randvec(self, x):
        # U = randskew
        raise NotImplementedError


    def transp(self, x1, x2, d):
        return d

    def exp(self, x, u, t=1):
        exptU = t*U
        for i in range(0, k):
            exptU[:,:,i] = la.expm(exptU[:,:,i])
        return multiprod(x, exptU)


    

    def pairmean(self, X, Y):
        '''
        Computes the intrinsic mean of X and Y, that is, a point that lies
        mid-way between X and Y on the geodesic arc joining them.
        '''
        V = self.log(X, Y)
        return self.exp(X, 0.5*V)