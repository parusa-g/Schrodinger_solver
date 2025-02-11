import sys
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
sys.path.append('../')
sys.path.append('../../PseudoTool/')
from Solve_schrodinger import SolveSchrodingerAtomic
from ReadPseudoXml import readxml
# =============================================================================
def Plot(x,y,xlabel='',ylabel='',title='',labels=[]):
    fig, ax = plt.subplots()
    
    ydim = y.ndim
    
    if ydim == 1:
        ax.plot(x, y)
    elif ydim == 2:
        for i in range(y.shape[1]):
            if len(labels) > 0:
                ax.plot(x, y[:,i], label=labels[i])
            else:
                ax.plot(x, y[:,i])
    
    if len(xlabel) > 0:
        ax.set_xlabel(xlabel)
        
    if len(ylabel) > 0:
        ax.set_ylabel(ylabel)
        
    if len(title) > 0:
        ax.set_title(title)
        
    if len(labels) > 0:
        ax.legend()
        
    plt.show()
# =============================================================================
def main():
    # Initialize the class
    calc = SolveSchrodingerAtomic(grid_type='mixed', xmin=0., xmax=100.,
                                  N=30, Np=7, Nu=10, Ng=10, rc=15.)
    
    # for plotting 
    rplot = np.linspace(0.0, 15., 300)
    
    
    # Example with only local potential 
    # ---------------------------------
    
    # This is all-electron potential from QE within atomic units 
    rr, Si_potential = np.loadtxt('Si_potential.dat', unpack=True)
    
    calc.boundPot(rr, Si_potential)
    
    # Compute the 3s state
    En, vn_3s = calc.getBound(l=0, n=3)
    print(f'Energy of 3s state: {En:.6f} Ha')
          
    # Compute the 3p state
    En, vn_3p = calc.getBound(l=1, n=3)
    print(f'Energy of 3p state: {En:.6f} Ha')
    
    # Compute the wave functions
    wfn_3s = calc.getWavefunc(vn_3s, rplot)
    wfn_3p = calc.getWavefunc(vn_3p, rplot)
    wfn = np.column_stack((wfn_3s, wfn_3p))
    
    Plot(rplot, wfn, xlabel='r [Bohr]', ylabel=r'$\psi(r)$', title='Silicon all-electron', labels=['3s', '3p'])
    
    
    # Example with pseudopotential
    # ----------------------------
    # Re-initialize the class, adjusting the xmax according to the potential data to be used (Vpsloc.dat)
    calc = SolveSchrodingerAtomic(grid_type='mixed', xmin=0., xmax=40.,
                                  N=30, Np=7, Nu=10, Ng=10, rc=15.)
        
    # Read the pseudopotential
    tree = ET.parse('Si.xml')
    root = tree.getroot()
    pp = readxml(root)
    
    # Get necessary data from the PP file
    Dij = pp.get_data('pp_dij')
    
    # Only the diagonal elements that matter within Hamann's scheme
    Dii = np.diag(Dij)
    
    # Vanderbilt projectors
    rmesh = pp.get_data('pp_rmesh')
    pp_beta = pp.get_data('pp_beta')
    
    # See the pp file for this one
    lbeta = [0, 0, 1, 1, 2, 2]
    
    # This is the total local potential: Vpsloc = Vloc_atom + V_H + V_xc
    rr, Vpsloc = np.loadtxt('Si_Vpsloc.dat', unpack=True)
    
    calc.boundPot(rr, Vpsloc, nonloc=True, Dii=Dii, rmesh=rmesh, beta=pp_beta, lbeta=lbeta)
    
    # Compute the 3s state
    # In the framework of pseudopotential, the lowest state being pseudized will be counted as 1s
    En, vn_3s = calc.getBound(l=0, n=1)
    print(f'Energy of 3s state: {En:.6f} Ha')
    
    # Compute the 3p state
    En, vn_3p = calc.getBound(l=1, n=2)
    print(f'Energy of 3p state: {En:.6f} Ha')
    
    # Compute the wave functions
    wfn_3s = calc.getWavefunc(vn_3s, rplot)
    wfn_3p = calc.getWavefunc(vn_3p, rplot)
    wfn = np.column_stack((np.real(wfn_3s), np.real(wfn_3p)))
    
    Plot(rplot, wfn, xlabel='r [Bohr]', ylabel=r'$\psi(r)$', title='Silicon pseudopotential', labels=['3s', '3p'])
# ===================================================================================================================
if __name__ == '__main__':
    main()

    
    
    
    
    
