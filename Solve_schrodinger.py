import sys
import numpy as np
import scipy.linalg as la
from scipy.special import legendre as leg
from scipy.special import eval_legendre as legval
from scipy.interpolate import make_interp_spline
sys.path.append('../PseudoTool/')
from ReadPseudoXml import parse
#==============================================================================
class SolveSchrodingerAtomic(object):
    '''
    Solves Schrodinger equation for bound-state atomic problem using 
    finite-element method - discrete variable representation (FEM-DVR).
    '''
#==============================================================================
    def __init__(self,grid_type,xmin,xmax,N=30,Np=7,Nu=10,Ng=10,rc=15.,pp_xml=''):
        self.grid = grid_type
        self.Ng = Ng
        
        # Determine the spacing
        # -----------------------
        if grid_type == 'linear':
            self.Ne = N
            self.xp = np.linspace(xmin,xmax,self.Ne+1)
            
        elif grid_type == 'upm':
            self.Ne = Np * Nu
            self.xp = self.upm(xmin,xmax,Np,Nu)
            
        elif grid_type == 'mixed':
            self.Ne = Np*Nu + N
            x1 = self.upm(xmin,rc,Np,Nu)[:-1]
            x2 = np.linspace(rc,xmax,N+1)
            self.xp = np.concatenate((x1,x2))
            
            
        self.Xg, self.Wg = self.getLobattos()
        self.xgrid = self.getAxis()
        self.fw, self.bw = self.getWeights()
        self.Kin = self.Kinetic()
        
        self.nonlocB = False
        if len(pp_xml) > 0:
            self.nonlocB = True
            
            pp = parse(pp_xml)
            Dij = pp.get_data('pp_dij')
            Dii = np.diag(Dij)
            rmesh = pp.get_data('pp_rmesh')
            beta = pp.get_data('pp_beta')
            lbeta = pp.llbeta
            
            self.init_nonloc(Dii, rmesh, beta, lbeta)
#==============================================================================
    def upm(self,xmin,xmax,Np,Nu):
        '''
        Universal power-mesh.
        '''
        h = (xmax - xmin) / (2**Np - 1)
        idx = np.arange(Np+1)
        pt = xmin + h*(2**idx - 1)
        
        ptd = np.roll(pt,-1)
        d = ptd[:Np] - pt[:Np]
        nd = np.arange(Nu) / Nu
        
        out = nd[None,:]*d[:,None] + pt[:Np,None]
        out = np.append(out,pt[-1])
        
        return out
#==============================================================================
    def getLobattos(self):
        '''
        Returns the Lobatto points and weights for the Gauss-Lobatto quadrature.
        '''
        LobattoPoints = leg(self.Ng).deriv(1).r
        LobattoPoints = sorted(LobattoPoints)
        LobattoPoints = np.insert(LobattoPoints, 0, -1.)
        
        # Lobatto weights
        val = self.Ng * (self.Ng+1) * legval(self.Ng, LobattoPoints)**2
        LobattoWeights = 2/val
        
        return LobattoPoints, LobattoWeights
#==============================================================================
    def getAxis(self):
        xp1 = np.roll(self.xp,-1)
        del1 = xp1[:self.Ne] - self.xp[:self.Ne]
        del2 = xp1[:self.Ne] + self.xp[:self.Ne]
        
        pts = del1[:,None]*self.Xg[None,:] + del2[:,None]
        pts = 0.5 * np.reshape(pts,-1)
        pts = np.append(pts,self.xp[-1])
        
        return pts
#==============================================================================
    def getWeights(self):
        xp1 = np.roll(self.xp,-1)
        dlt = xp1[:self.Ne] - self.xp[:self.Ne]

        # Forward weights
        w1 = 0.5 * self.Wg[None,:] * dlt[:,None]
        # to avoid divide-by-zero
        w1 = np.append(w1,1.)

        # Backward weights
        dlt = np.insert(dlt,0,1.)
        w2 = dlt/ self.Ng/ (self.Ng + 1)

        return w1, w2
#==============================================================================
    def LobattoPoly(self,i,m,x):
        '''
        Lobatto polynomial.
        '''
        
        val = 0.
        if i>=0 and i<self.Ne-1:
            if i!=0 or m!=0:
                if x>=self.xgrid[i * self.Ng] and x<self.xgrid[(i+1) * self.Ng]:
                    val = 1.
                    for j in range(self.Ng+1):
                        if j!=m:
                            val *= x - self.xgrid[i*self.Ng + j]
                            val /= self.xgrid[i*self.Ng + m] - self.xgrid[i*self.Ng + j]
                        
        elif i==self.Ne-1 and (m>=0 and m<=self.Ng):
            if x>=self.xgrid[i * self.Ng]:
                val = 1.
                for j in range(self.Ng+1):
                    if j!=m:
                        val *= x - self.xgrid[i*self.Ng + j]
                        val /= self.xgrid[i*self.Ng + m] - self.xgrid[i*self.Ng + j]

        return val
#==============================================================================
    def LobattoBasis(self,i,m,x):
        val = 0.
        if m==0:
            val = self.LobattoPoly(i-1,self.Ng,x) + self.LobattoPoly(i,0,x)
            val /= np.sqrt(self.bw[i] + self.fw[i*self.Ng])
            
        elif m==self.Ng:
            val = self.LobattoPoly(i,self.Ng,x) + self.LobattoPoly(i+1,0,x)
            val /= np.sqrt(self.bw[i+1] + self.fw[(i+1)*self.Ng])
            
        elif m>0 and m<self.Ng:
            val = self.LobattoPoly(i,m,x)
            val /= np.sqrt(self.fw[i*self.Ng + m])
            
        return val
#==============================================================================
    def derLobatto(self,i,m1,m2):
        '''
        Calculating the first derivative of Lobatto polynomial.
        '''
        
        val = 0.
        if i>=0 and i<self.Ne:
            if (i!=0 or m1!=0) and (i!=self.Ne or m1!=0):
                if m1!=m2:
                    val = 1.
                    for j in range(self.Ng+1):
                        if j!=m1 and j!=m2:
                            val *= self.xgrid[i*self.Ng + m2] - self.xgrid[i*self.Ng + j]
                            val /= self.xgrid[i*self.Ng + m1] - self.xgrid[i*self.Ng + j]
                            
                    val /= self.xgrid[i*self.Ng + m1] - self.xgrid[i*self.Ng + m2]
                    
                else:
                    if m1==0:
                        val = -1./ (2 * self.fw[i*self.Ng])
                        
                    elif m1==self.Ng:
                        val = 1./ (2 * self.bw[i+1])
                        
        return val
#==============================================================================
    def derivmat(self,i):
        fderiv = np.vectorize(self.derLobatto)
        m1 = np.arange(self.Ng+1)
        m2 = np.copy(m1)
        
        mat = fderiv(i,m1[:,None],m2[None,:])
        
        return mat
#==============================================================================
    def ttilde(self,i):
        w = np.copy(self.fw[i*self.Ng: (i+1)*self.Ng])
        w = np.append(w,self.bw[i+1])
        D = self.derivmat(i)
        
        mat = np.einsum('ij,kj,j->ik',D,D,w)
        
        return mat
#==============================================================================
    def Kinetic(self):
        '''
        Calculate the kinetic energy matrix elements.
        '''
        n = self.Ne * self.Ng
        T = np.zeros((n,n))
        
        for i in range(self.Ne):
            tm = self.ttilde(i)
            tmm1 = self.ttilde(i-1)
            
            # i1 = i2
            # -------------------
            
            # m1>0, m2>0
            t = np.copy(tm[1: self.Ng, 1: self.Ng])
            w1 = np.copy(self.fw[i*self.Ng+1: (i+1)*self.Ng])
            w2 = np.copy(w1)
            w = np.sqrt(w1[:,None]*w2[None,:])
            T[i*self.Ng+1: (i+1)*self.Ng, i*self.Ng+1: (i+1)*self.Ng] = 0.5 * t / w
            
            # m1=0, m2>0
            t = np.copy(tm[0, 1: self.Ng])
            w1 = np.copy(self.fw[i*self.Ng+1: (i+1)*self.Ng])
            w2 = self.bw[i] + self.fw[i*self.Ng]
            w = np.sqrt(w1*w2)
            T[i*self.Ng, i*self.Ng+1: (i+1)*self.Ng] = 0.5 * t / w
            
            # m1>0, m2=0
            t = np.copy(tm[1: self.Ng, 0])
            T[i*self.Ng+1: (i+1)*self.Ng, i*self.Ng] = 0.5 * t / w
            
            # m1=m2=0
            t = np.copy(tm[0,0] + tmm1[-1,-1])
            w = self.bw[i] + self.fw[i*self.Ng]
            T[i*self.Ng, i*self.Ng] = 0.5 * t / w
            
            # i1 = i2+1
            # --------------------
            
            if i>0:
                
                # m1=0, m2>0
                t = np.copy(tmm1[-1, 1: self.Ng])
                w1 = np.copy(self.fw[(i-1)*self.Ng+1: i*self.Ng])
                w2 = np.copy(self.bw[i] + self.fw[i*self.Ng])
                w = np.sqrt(w1*w2)
                T[i*self.Ng, (i-1)*self.Ng+1: i*self.Ng] = 0.5 * t / w
                
                # m1=m2=0
                t = np.copy(tmm1[-1,0])
                w1 = np.copy(self.bw[i] + self.fw[i*self.Ng])
                w2 = np.copy(self.bw[i-1] + self.fw[(i-1)*self.Ng])
                w = np.sqrt(w1*w2)
                T[i*self.Ng, (i-1)*self.Ng] = 0.5 * t / w
                
            
            # i1 = i2-1
            # --------------------
            
            if i<self.Ne-1:
                
                # m1>0, m2=0
                t = np.copy(tm[1:self.Ng, -1])
                w1 = np.copy(self.fw[i*self.Ng+1: (i+1)*self.Ng])
                w2 = np.copy(self.bw[i+1] + self.fw[(i+1)*self.Ng])
                w = np.sqrt(w1*w2)
                T[i*self.Ng+1: (i+1)*self.Ng, (i+1)*self.Ng] = 0.5 * t / w
                
                # m1=m2=0
                t = np.copy(tm[0,-1])
                w1 = np.copy(self.bw[i] + self.fw[i*self.Ng])
                w2 = np.copy(self.bw[i+1] + self.fw[(i+1)*self.Ng])
                w = np.sqrt(w1*w2)
                T[i*self.Ng, (i+1)*self.Ng] = 0.5 * t / w
                
        return T[1:, 1:]
#==============================================================================
    def init_nonloc(self,D,rmesh,beta,lbeta):
        '''
        Initialize variables for using a non-local potential.
        '''
        self.D = D
        self.rmesh = rmesh
        self.beta = beta
        self.lbeta = lbeta
#==============================================================================
    def get_Vnl(self,l):
        '''
        Calculate the non-local potential matrix elements.
        '''
        nproj = self.beta.shape[-1]
        N = len(self.xgrid[1:-1])
        chjj = np.zeros((N,N), dtype=complex)
        
        for i in range(nproj):
            if self.lbeta[i] == l:
                splr = make_interp_spline(self.rmesh, np.real(self.beta[:,i]))
                spli = make_interp_spline(self.rmesh, np.imag(self.beta[:,i]))

                beta_r = splr(self.xgrid[1:-1])
                ch_re = [beta_r for i in range(N)]
                ch_re = np.array(ch_re)

                beta_i = spli(self.xgrid[1:-1])
                ch_im = [beta_i for i in range(N)]
                ch_im = np.array(ch_im)

                chj = ch_re + 1j*ch_im
                chjj += self.D[i] * chj * np.conj(chj.T)

        w = np.copy(self.fw)
        w = np.sqrt(w)
        for i in range(self.Ne):
            w[i*self.Ng] = np.sqrt(self.bw[i] + self.fw[i*self.Ng])

        wj = [w[1:-1] for i in range(N)]
        wj = np.array(wj)
        wjj = wj * wj.T
        
        if wjj.shape != chjj.shape:
            raise ValueError('Something wrong with the dimension of the \
                              weight and/or projector matrices. ')

        Vnl = wjj * chjj

        return Vnl
#==============================================================================
    def boundPot(self,rr,Vloc):
        '''
        Initialize the bound-state potential.
        rr     : radial grid for Vloc
        Vloc   : local potential
        '''
    
        self.Vrb = make_interp_spline(rr, Vloc, k=3)
#==============================================================================
    def Vlr(self,l):
        Vr = self.Vrb(self.xgrid[1:-1])
        Vl = 0.5 * l*(l+1) / self.xgrid[1:-1]**2
        Vlr = Vr + Vl
        
        return Vlr 
#==============================================================================
    def getBound(self,l,n=None):
        Vlr = np.diag(self.Vlr(l))
        Ham = self.Kin + Vlr
        
        if self.nonlocB == True:
            Ham = Ham + self.get_Vnl(l)

        E, V = la.eigh(Ham)
        
        if n != None:
            k = n - l - 1
            if k >= 0:
                E = E[k]
                V = V[:,k]
            else: 
                raise ValueError('wrong n value')
        
        return E, V
#==============================================================================
    def Psi(self,eigv,x):
        '''
        Calculate the wave function at a given point x.
        '''
        
        chi = [[self.LobattoBasis(i,m,x) for m in range(self.Ng)] for i in range(self.Ne)]
        chi = np.array(chi).reshape(-1)

        psi = np.dot(eigv, chi[1:])

        return psi
#==============================================================================
    def getWavefunc(self,eigv,rr):
        '''
        Calculate the wave function on a given radial grid.
        '''
        
        psi = [self.Psi(eigv,x) for x in rr]
        psi = np.array(psi)
        
        psi = np.real(psi)
        irmax = np.argmax(np.abs(psi))
        psi *= np.sign(psi[irmax])

        return psi
#==============================================================================