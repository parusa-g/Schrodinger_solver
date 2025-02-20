import sys
import numpy as np
import confinement
from scipy.optimize import minimize_scalar as minimize
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
    def __init__(self,grid_type,xmin,xmax,confinement,rnew,rr,Vr,pp_xml):
        '''
        grid_type   : [str]    type of grid
        xmin        : [float]  minimum of the grid
        xmax        : [float]  maximum of the grid
        confinement : [str]    type of confinement potential
        pp_xml      : [str]    pseudopotential file in xml format
        rnew        : [array]  new radial grid
        rr          : [array]  original radial grid
        Vr          : [array]  original potential
        '''
        self.confinement = confinement
        self.rnew = rnew
        self.rloc = rr
        self.Vloc = Vr
        
        solver.__init__(self, grid_type, xmin, xmax, pp_xml=pp_xml)
    
        pseudo = parse(pp_xml)
        rmesh = pseudo.get_data('pp_rmesh')
        pseudowf = pseudo.get_data('pp_pswfc')
        
        self.pseudowf_spl = make_interp_spline(rmesh, pseudowf, k=3)
        self.llpseudowf = pseudo.llpswfc
    # ----------------------------------------------------------------------------
    def GetVnew(self,alpha,params=[]):
        '''
        Get the new potential on the new grid rnew.
        alpha : [float] confinement parameter
        params: [list]  additional parameters for the confinement potential
        '''
        
        if self.confinement == 'quadratic':
            Vnew = confinement.ConfineQuadratic(alpha, self.rnew, self.rr, self.Vr)
            
        elif self.confinement == 'quadratic+der1':
            Vnew = confinement.ConfineQuadratic(alpha, self.rnew, self.rr, self.Vr, der1=True)
            
        elif self.confinement == 'quartic':
            if len(params) > 0:
                V0 = params[0]
                # alpha in this case is the confinement radius
                Vnew = confinement.ConfineAsympQuartic(alpha, V0, self.rnew, self.rr, self.Vr)
            else:
                raise ValueError('Provide the asymptotic value of the confinement potential')
            
        elif self.confinement == 'additive':
            if len(params) > 0:
                n = params[0]
                Vnew = confinement.ConfineAddOrder(alpha, self.rnew, self.rr, self.Vr, n)
            else:
                raise ValueError('Provide the order of the confinement potential')
            
        elif self.confinement == 'one_over_r':
            if len(params) > 1:
                rc = params[0]
                n = params[1]
                Vnew = confinement.ConfinePolyOneOverR(alpha, rc, self.rnew, self.rr, self.Vr, n)
            else:
                raise ValueError('Provide the confinement radius and \
                                 the order of the confinement potential')
    
        return Vnew
    # ----------------------------------------------------------------------------
    def LossFunction(self,alpha,ll,params=[]):
        '''
        Loss function to be minimized: mean-squared error between the bound-state from the confined
        potential and the corresponding pseudo-wavefunction. Optimize for angular momentum channel ll.
        alpha : [float] confinement parameter
        params: [list]  additional parameters for the confinement potential
        '''
        
        Vnew = self.GetVnew(alpha, params)
        
        solver.boundPot(self.rnew, Vnew)
        _, vn = solver.getBound(l=ll, n=ll+1)
        wfn = solver.getWavefunc(vn, self.rnew)
        
        mse = np.mean((self.pseudowf_spl(self.rnew) - wfn)**2)
        
        return mse
    # ----------------------------------------------------------------------------
    def OptimizeConfinement(self,ll,params=[]):
        '''
        Optimize the confinement parameter for a given angular momentum channel ll.
        ll     : [int]   angular momentum channel
        params : [list]  additional parameters for the confinement potential
        '''
        
        res = minimize(self.LossFunction, args=(ll,params), bounds=(0., None))
        
        if res.succsess:
            return res.x
        else:
            raise ValueError('Optimization failed. Reason: {}'.format(res.message))
# =================================================================================
        
        
        