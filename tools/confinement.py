import numpy as np
from scipy.interpolate import make_interp_spline
# ==============================================================================
def ConfineQuadratic(alpha,rnew,rr,V,der1=False):
   '''
   Confinement r >= R:
                V(r) = alpha * (r - R)^2 + V(R),                  if der1 = False
                V(r) = alpha * (r - R)^2 + V(R) + dV * (r - R),   if der1 = True
   
   R is the last grid point of the original potential.
   dV is the first derivative of the original potential at the last point, calculated 
   with a central difference scheme.
      
   alpha : [float]  curvature of the confinement potential
   rnew  : [array]  new radial grid
   rr    : [array]  original radial grid
   V     : [array]  original potential
   der1  : [bool]   if True, continuity up to the first derivative is enforced
   '''
   spl = make_interp_spline(rr, V, k=3)
   R = rr[-1]
   
   Vnew = np.zeros_like(rnew)

   Ir, = np.where(rnew <= R)
   Vnew[Ir] = spl(rnew[Ir])
   
   Ir, = np.where(rnew > R)
   Vnew[Ir] = alpha * (rnew[Ir] - R)**2 + V[-1] 
   
   if der1:
       h = 1e-6
       dV = (spl(R+h) - spl(R-h)) / (2*h)
       Vnew[Ir] += dV * (rnew[Ir] - R)

   return Vnew
# ====================================================================================================
def GetVnewQuartic(rnew,rc,R,V0,a0,spl,h=1e-6):
   '''
    Get the new potential with fourth order polynomial with continuity up to the first derivative
    at the last point of the original potential and asymptotically goes to V0 at rc with first
    and the second derivatives at rc being zero.
    rnew: new radial grid
    rc  : confinement radius
    R   : last point of the original potential
    V0  : asymptotic value of the confinement potential
    a0  : first derivative of the original potential at R
    spl : spline interpolator for the original potential
    h   : small number for numerical differentiation
   ''' 
   v = rc - R
   a1 = (spl(R+h) - spl(R-h)) / (2*h)
   
   b = np.zeros(3)
   b[0] = V0 - a0 - a1*v
   b[1] = -a1
   
   A = np.zeros([3,3])
   A[0,0] = v**2; A[0,1] = v**3   ; A[0,2] = v**4
   A[1,0] = 2*v ; A[1,1] = 3*v**2 ; A[1,2] = 4*v**3
   A[2,0] = 2   ; A[2,1] = 6*v    ; A[2,2] = 12*v**2
   
   x = np.linalg.solve(A,b)
   
   Vnew = a0 + a1*(rnew-R) + x[0]*(rnew - R)**2 + \
             x[1]*(rnew - R)**3 + x[2]*(rnew - R)**4
   
   return Vnew
# ====================================================================================================
def ConfineAsympQuartic(rc,V0,rnew,rr,V):
   '''
   Confinement r >= R:
                V(r) = a0 + a1*(r - R) + a2*(r - R)^2 + a3*(r - R)^3 + a4*(r - R)^4,   if r <= rc
                V(r) = V0,                                                             if r > rc
                
   R is the last grid point of the original potential.
   a0, a1, a2, a3, a4 are determined by the continuity of the potential at R up to the first derivative,
   asymptotic value at rc (V0), and the first and second derivatives at rc being zero.
    
   rc  : [float]  confinement radius
   V0  : [float]  asymptotic value of the confinement potential
   rnew: [array]  new radial grid
   rr  : [array]  original radial grid
   V   : [array]  original potential
   '''
   spl = make_interp_spline(rr, V, k=3)
   R = rr[-1]
   
   Vnew = np.zeros_like(rnew)

   Ir, = np.where(rnew <= R)
   Vnew[Ir] = spl(rnew[Ir])

   Ir, = np.where((rnew > R) & (rnew <= rc))
   Vnew[Ir] = GetVnewQuartic(rnew[Ir], rc, R, V0, V[-1], spl)
              
   Ir, = np.where(rnew > rc)
   Vnew[Ir] = V0

   return Vnew
# ====================================================================================================
def ConfineAddOrder(alpha,rnew,rr,V,n=4):
   '''
   Confinement r > R:
                    V(r) = alpha * (r - R)^n + V(R) + dV * (r - R)
                    
   R is the last grid point of the original potential.
   dV is the first derivative of the original potential at the last point, calculated
   with a central difference scheme.
   
   alpha : [float]   scaling factor of the confinement potential
   rnew  : [array]   new radial grid
   rr    : [array]   original radial grid
   V     : [array]   original potential
   n     : [integer] order of the confinement potential
   '''

   spl = make_interp_spline(rr, V, k=3)
   R = rr[-1]
   
   Vnew = np.zeros_like(rnew)

   Ir, = np.where(rnew <= R)
   Vnew[Ir] = spl(rnew[Ir])
   
   Ir, = np.where(rnew > R)
   
   h = 1e-6
   dV = (spl(R+h) - spl(R-h)) / (2*h)

   Vnew[Ir] = alpha * (rnew[Ir] - R)**n + V[-1] + dV * (rnew[Ir] - R)
   
   return Vnew
# ====================================================================================================
def GetCoeffVnewOneOverR(R,Vlast,dVlast):
    '''
    Get the coefficients for the intermediate zone potential with continuity up to the first derivative
    at the last point of the original potential. The potential is of the form,
                               
                                V(r) = a1/r + a2/r^2
                                
    R     : [float]  last point of the original potential
    Vlast : [float]  value of the original potential at R
    dVlast: [float]  first derivative of the original potential at R
    '''
    
    A = np.zeros([2,2])
    b = np.zeros(2)
    
    A[0,0] = 1.; A[0,1] = 1./R
    A[1,0] = 1.; A[1,1] = 2./R**2
    
    b[0] = Vlast * R
    b[1] = -dVlast * R**2
    
    x = np.linalg.solve(A,b)
    
    return x
# ====================================================================================================
def ConfinePolyOneOverR(alpha,rc,rnew,rr,V,n=4):
    '''
    Confinement r > R:
                    V(r) = x1/r + x2/r^2,   if r <= rc
                    V(r) = alpha * (r - rc)^n + a0 + a1*(r - rc),   if r > rc
    
    R is the last grid point of the original potential.
    x1, x2 are determined by the continuity of the potential at R up to the first derivative.
    a0, a1 are determined by the continuity of the potential at rc up to the first derivative.
    
    alpha : scaling factor of the confinement potential
    rc    : confinement radius
    rnew  : new radial grid
    rr    : original radial grid
    V     : original potential
    n     : order of the confinement potential
    '''
    
    Vnew = np.zeros_like(rnew) 
    
    R = rr[-1]
    spl = make_interp_spline(rr,V,k=3)
    
    h = 1e-6
    dV = (spl(R+h) - spl(R-h)) / (2*h)
    
    Ir, = np.where(rnew <= R)
    Vnew[Ir] = spl(rnew[Ir])
    
    Ir, = np.where((rnew > R) & (rnew <= rc))
    x = GetCoeffVnewOneOverR(R, V[-1], dV)
    Vnew[Ir] = x[0]/rnew[Ir] + x[1]/rnew[Ir]**2
    
    Ir, = np.where(rnew > rc)
    a0 = x[0]/rc + x[1]/rc**2
    a1 = -x[0]/rc**2 - 2*x[1]/rc**3
    Vnew[Ir] = alpha * (rnew[Ir] - rc)**n + a0 + a1*(rnew[Ir] - rc)
    
    return Vnew
# ====================================================================================================    