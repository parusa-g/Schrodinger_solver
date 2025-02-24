import numpy as np
from scipy.interpolate import make_interp_spline
from fornberg import fornberg_weights
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
       m = 1
       w = fornberg_weights(R, rr, m)
       dV = np.dot(V, w[:,m])
       Vnew[Ir] += dV * (rnew[Ir] - R)

   return Vnew
# ====================================================================================================
def GetVnewQuartic(rnew,rc,R,V0,a0,a1):
   '''
    Get the new potential with fourth order polynomial with continuity up to the first derivative
    at the last point of the original potential and asymptotically goes to V0 at rc with first
    and the second derivatives at rc being zero.
    rnew: new radial grid
    rc  : confinement radius
    R   : last grid point of the original potential
    V0  : asymptotic value of the confinement potential
    a0  : last point of the original potential
    a1  : first derivative of the original potential at the last point
   ''' 
   v = rc - R
   
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
   w = fornberg_weights(R, rr, 1)
   dV = np.dot(V, w[:,1])
   Vnew[Ir] = GetVnewQuartic(rnew[Ir], rc, R, V0, V[-1], dV)
              
   Ir, = np.where(rnew > rc)
   Vnew[Ir] = V0

   return Vnew
# ====================================================================================================
def ConfineAddOrder(alpha,rnew,rr,V,n=4):
   '''
   Confinement r > R:
                    V(r) = alpha * (r - R)^n + V(R) + dV * (r - R)
                    
   R is the last grid point of the original potential.
   dV is the first derivative of the original potential at the last point
   
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
   
   w = fornberg_weights(R, rr, 1)
   dV = np.dot(V, w[:,1])

   Vnew[Ir] = alpha * (rnew[Ir] - R)**n + V[-1] + dV * (rnew[Ir] - R)
   
   return Vnew
# ====================================================================================================
def GetCoeffVnewOneOverR(R,dVlast):
    '''
    Get the coefficients for the intermediate zone potential with continuity up to the 2nd derivative
    at the last point of the original potential. The potential is of the form,
                               
                                V(r) = \sum_{i=0}^{n} a_i/r^i
                                
    R     : [float]  last point of the original potential
    dVlast: [array]  derivatives of the original potential at R, starting from zero-th order (Vlast)
    '''
    
    n = len(dVlast)
    if n > 3:
        raise ValueError('Derivative order must be less than 3')
    
    A = np.zeros([3, 3])
    A[0,0] = 1 / R ; A[0,1] = 1 / R**2; A[0,2] = 1 / R**3
    A[1,0] = 1 / R ; A[1,1] = 2 / R**2; A[1,2] = 3 / R**3
    A[2,0] = 2 / R ; A[2,1] = 6 / R**2; A[2,2] = 12/ R**3
    
    b = np.zeros(3)
    b[0] = np.copy(dVlast[0])
    b[1] = -R * np.copy(dVlast[1])
    b[2] = R**2 * np.copy(dVlast[2])
    
    if n == 2:
        A = A[:2,:2]
        b = b[:2]
        
    x = np.linalg.solve(A,b)
    
    return x
# ====================================================================================================
def ConfinePolyOneOverR(alpha,rc,rnew,rr,V,n=4):
    '''
    Confinement r > R:
                    V(r) = x1/r + x2/r^2 + x3/r^3,             if r <= rc
                    V(r) = alpha * (r - rc)^n + a0 + a1*(r - rc) + a2*(r - rc)^2,   if r > rc
    
    R is the last grid point of the original potential.
    x1, x2 are determined by the continuity of the potential at R up to the second derivative.
    a0, a1 are determined by the continuity of the potential at rc up to the second derivative.
    
    alpha : scaling factor of the confinement potential
    rc    : confinement radius
    rnew  : new radial grid
    rr    : original radial grid
    V     : original potential
    n     : order of the confinement potential
    '''
    
    if n < 3: 
        raise ValueError('Order of the potential must be greater than 2')
    
    Vnew = np.zeros_like(rnew) 
    
    R = rr[-1]
    spl = make_interp_spline(rr,V,k=3)
    
    w = fornberg_weights(R, rr, 2)
    dV = np.dot(V, w[:,1])
    d2V = np.dot(V, w[:,2])
    
    Ir, = np.where(rnew <= R)
    Vnew[Ir] = spl(rnew[Ir])
    
    Ir, = np.where((rnew > R) & (rnew <= rc))
    x = GetCoeffVnewOneOverR(R, [V[-1], dV, d2V])
    Vnew[Ir] = x[0]/rnew[Ir] + x[1]/rnew[Ir]**2 + x[2]/rnew[Ir]**3
    
    Ir, = np.where(rnew > rc)
    a0 = x[0]/rc + x[1]/rc**2 + x[2]/rc**3
    a1 = -x[0]/rc**2 - 2*x[1]/rc**3 - 3*x[2]/rc**4
    a2 = 2*x[0]/rc**3 + 6*x[1]/rc**4 + 12*x[2]/rc**5
    Vnew[Ir] = alpha * (rnew[Ir] - rc)**n + a0 + a1*(rnew[Ir] - rc) \
                + a2*(rnew[Ir] - rc)**2
    
    return Vnew
# ==================================================================================================== 
def ConfinePolyOneOverRwithFermi(alpha,rc,V0,rnew,rr,V):
   
    Vnew = np.zeros_like(rnew)
    VFermi = np.zeros_like(rnew)
   
    VFermi = V0 / (1 + np.exp((rc - rnew)/ alpha))
   
    spl = make_interp_spline(rr, V, k=3)
    R = rr[-1]
   
    w = fornberg_weights(R, rr, 2)
    dV = np.dot(V, w[:,1])
    d2V = np.dot(V, w[:,2])
   
    Ir, = np.where(rnew <= R)
    Vnew[Ir] = spl(rnew[Ir])
   
    Ir, = np.where(rnew > R)
    x = GetCoeffVnewOneOverR(R, [V[-1], dV, d2V])
    Vnew[Ir] = x[0]/rnew[Ir] + x[1]/rnew[Ir]**2 + x[2]/rnew[Ir]**3
   
    Vnew = Vnew + VFermi
   
    return Vnew
# ====================================================================================================    