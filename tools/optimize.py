import sys
import numpy as np
import confinement
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
sys.path.append('../')
sys.path.append('../../PseudoTool/')
from ReadPseudoXml import parse
from Solve_schrodinger import SolveSchrodingerAtomic as solver
# ==============================================================================
class OptimizeConfinement(solver):
    '''
    Optimize the confinement parameters for a given atomic potential.
    '''
    def __init__(self,grid_type,xmin,xmax,confinement,rpot,rloc,Vloc,pp_xml):
        '''
        grid_type   : [str]    type of grid
        xmin        : [float]  minimum of the grid
        xmax        : [float]  maximum of the grid
        confinement : [str]    type of confinement potential
        pp_xml      : [str]    pseudopotential file in xml format
        rpot        : [array]  new radial grid for the confined potential
        rloc        : [array]  original radial grid
        Vloc        : [array]  original potential
        '''
        self.confinement = confinement
        
        # Original potential data
        self.rloc = rloc
        self.Vloc = Vloc
        
        # Schrodinger solver        
        solver.__init__(self, grid_type, xmin, xmax, pp_xml=pp_xml)
    
        pseudo = parse(pp_xml)
        
        # Grid for the wave function
        self.rmesh = pseudo.get_data('pp_rmesh')
        self.pseudowf = pseudo.get_data('pp_pswfc')
        self.llpseudowf = np.array(pseudo.llpswfc)
        
        # Grid for the confined potential 
        self.rpot = rpot
    # ----------------------------------------------------------------------------
    def GetVnew(self,alpha,params=[]):
        '''
        Get the new potential on the new grid rpot.
        alpha : [float] confinement parameter
        params: [list]  additional parameters for the confinement potential
        '''
        
        if self.confinement == 'quadratic':
            Vnew = confinement.ConfineQuadratic(alpha, self.rpot, self.rloc, self.Vloc)   
                     
        elif self.confinement == 'quadratic+der1':
            Vnew = confinement.ConfineQuadratic(alpha, self.rpot, self.rloc, self.Vloc, der1=True)
            
        elif self.confinement == 'quartic':
            if len(params) > 0:
                V0 = params[0]
                # alpha in this case is the confinement radius
                Vnew = confinement.ConfineAsympQuartic(alpha, V0, self.rpot, self.rloc, self.Vloc)
            else:
                raise ValueError('Provide the asymptotic value of the confinement potential')
            
        elif self.confinement == 'additive':
            if len(params) > 0:
                n = params[0]
                Vnew = confinement.ConfineAddOrder(alpha, self.rpot, self.rloc, self.Vloc, n)
            else:
                raise ValueError('Provide the order of the confinement potential')
            
        elif self.confinement == 'one_over_r':
            if len(params) > 1:
                rc = params[0]
                n = params[1]
                Vnew = confinement.ConfinePolyOneOverR(alpha, rc, self.rpot, self.rloc, self.Vloc, n)
            else:
                raise ValueError('Provide the confinement radius and \
                                 the order of the confinement potential')
                
        elif self.confinement == 'one_over_r+Fermi':
            if len(params) > 1:
                rc = params[0]
                V0 = params[1]
                Vnew = confinement.ConfinePolyOneOverRwithFermi(alpha, rc, V0, self.rpot, self.rloc, self.Vloc)
            else:
                raise ValueError('Provide the confinement radius and \
                                  the asymptotic value of the confinement potential')    
            
    
        return Vnew
    # ----------------------------------------------------------------------------
    def GetConfinedBoundState(self,alpha,l,n=None,params=[]):
        '''
        Get the bound-state for a given angular momentum channel l.
        alpha : [float] confinement parameter
        l     : [int]   angular momentum channel
        n     : [int]   principal quantum number
        params: [list]  additional parameters for the confinement potential
        '''
        
        Vnew = self.GetVnew(alpha, params)
        
        super().boundPot(self.rpot, Vnew)
        if n != None:
            _, vn = super().getBound(l=l, n=n)
            wfn = super().getWavefunc(vn, self.rmesh)
        else:
            _, vn = super().getBound(l=l, n=l+1)
            wfn = super().getWavefunc(vn, self.rmesh)
        
        return wfn
    # ----------------------------------------------------------------------------
    def LossFunction(self,alpha,l,params=[]):
        '''
        Loss function to be minimized: mean-squared error between the bound-state from the confined
        potential and the corresponding pseudo-wavefunction. Optimize for angular momentum channel l.
        alpha : [array] confinement parameter
        params: [list]  additional parameters for the confinement potential
        '''
        
        wfn = self.GetConfinedBoundState(alpha[0],l,params=params)
        
        # I expect only one pseudo-wavefunction per l channel
        il, = np.where(self.llpseudowf == l)[0]
        mse = np.mean((self.pseudowf[:,il] - wfn)**2)
        
        return mse
    # ----------------------------------------------------------------------------
    def Run(self,l,alpha0,params=[]):
        '''
        Optimize the confinement parameter for a given angular momentum channel l.
        l      : [int]   angular momentum channel
        alpha0 : [array] initial guess for the confinement parameter
        params : [list]  additional parameters for the confinement potential
        '''
        
        res = minimize(self.LossFunction, alpha0, args=(l,params))
        
        if res.success:
            return res.x
        else:
            raise ValueError('Optimization failed. Reason: {}'.format(res.message))
# =================================================================================
        
        
        