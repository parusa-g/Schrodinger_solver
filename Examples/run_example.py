import sys
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
sys.path.append('../')
sys.path.append('../../PseudoTool')
from Solve_schrodinger import SolveSchrodingerAtomic
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
    
    calc.InitPotential(rr, Si_potential)
    
    # Compute the 3s state
    En, vn_3s = calc.GetBound(l=0, n=3)
    print(f'Energy of 3s state: {En:.6f} Ha')
          
    # Compute the 3p state
    En, vn_3p = calc.GetBound(l=1, n=3)
    print(f'Energy of 3p state: {En:.6f} Ha')
    
    # Compute the wave functions
    wfn_3s = calc.GetWavefunc(vn_3s, rplot)
    wfn_3p = calc.GetWavefunc(vn_3p, rplot)
    wfn = np.column_stack((wfn_3s, wfn_3p))
    
    Plot(rplot, wfn, xlabel='r [Bohr]', ylabel=r'$\psi(r)$', title='Silicon all-electron', labels=['3s', '3p'])

    print('\n')

    # Compute a scattering state
    E = 1.0 # Hartree

    print(f'Computing scattering state at E = {E:.2f} Ha')
    vn_s = calc.GetScatt(E, l=0)
    vn_p = calc.GetScatt(E, l=1)
    vn_d = calc.GetScatt(E, l=2)

    wfn_s = calc.GetWavefunc(vn_s, rplot, bound=False, l=0)
    wfn_p = calc.GetWavefunc(vn_p, rplot, bound=False, l=1)
    wfn_d = calc.GetWavefunc(vn_d, rplot, bound=False, l=2)

    # Only pack the real part
    wfn_scatt = np.column_stack((wfn_s.real, wfn_p.real, wfn_d.real))

    Plot(rplot, wfn_scatt, xlabel='r [Bohr]', ylabel=r'Real [$\psi(r)$]', title='Silicon scattering states', labels=['s', 'p', 'd'])

    print('\n')

    # Compute the scattering phase
    # Should start from Ek > 0.0
    KinE = np.linspace(0.1, 6.0, 100)

    print('Computing scattering phase')
    smooth = True
    phase_s = calc.GetScatteringPhase(KinE, l=0, smooth=smooth)
    phase_p = calc.GetScatteringPhase(KinE, l=1, smooth=smooth)
    phase_d = calc.GetScatteringPhase(KinE, l=2, smooth=smooth)
    phase = np.column_stack((phase_s, phase_p, phase_d))

    Plot(KinE, phase, xlabel='Kinetic energy [Ha]', ylabel=r'Phase [$\pi$]', title='Silicon scattering phase', labels=['s', 'p', 'd'])

    print('\n')

    # Compute the radial integral
    smooth = False
    print('Computing radial integrals for 3s')
    radint_3s_abs, radint_3s_angle = calc.GetRadialIntegral(KinE, n=3, l=0, verbose='low', store_type='abs-angle', smooth_phase=smooth)

    print('Computing radial integrals for 3p')
    radint_3p_abs, radint_3p_angle = calc.GetRadialIntegral(KinE, n=3, l=1, verbose='low', store_type='abs-angle', smooth_phase=smooth)

    radint_abs = np.column_stack((radint_3s_abs, radint_3p_abs))
    radint_angle = np.column_stack((radint_3s_angle, radint_3p_angle))
    
    Plot(KinE, radint_abs, xlabel='Kinetic energy [Ha]', ylabel=r'Absolute value', title='Silicon radial integrals', 
         labels=[r'$s \rightarrow p$', r'$p \rightarrow s$', r'$p \rightarrow d$'])
    Plot(KinE, radint_angle, xlabel='Kinetic energy [Ha]', ylabel=r'Phase [$\pi$]', title='Silicon radial integrals phase', 
         labels=[r'$s \rightarrow p$', r'$p \rightarrow s$', r'$p \rightarrow d$'])

    print('\n')

    # Example with pseudopotential
    # ----------------------------
    # Re-initialize the class. Note also the pseudopotential xml file.
    calc = SolveSchrodingerAtomic(grid_type='mixed', xmin=0., xmax=100.,
                                  N=30, Np=7, Nu=10, Ng=10, rc=15.,
                                  pp_xml='Si.xml')
        
    # This is the total local potential: Vpsloc = Vloc_atom + V_H + V_xc
    rr, Vpsloc = np.loadtxt('Si_Vpsloc.dat', unpack=True)
    
    calc.InitPotential(rr, Vpsloc)
    
    # Compute the 3s state
    # In the framework of pseudopotential, the lowest state being pseudized will be counted as 1s
    En, vn_3s = calc.GetBound(l=0, n=1)
    print(f'Energy of 3s state: {En:.6f} Ha')
    
    # Compute the 3p state
    En, vn_3p = calc.GetBound(l=1, n=2)
    print(f'Energy of 3p state: {En:.6f} Ha')
    
    # Compute the wave functions
    wfn_3s = calc.GetWavefunc(vn_3s, rplot)
    wfn_3p = calc.GetWavefunc(vn_3p, rplot)
    wfn = np.column_stack((wfn_3s, wfn_3p))
    
    Plot(rplot, wfn, xlabel='r [Bohr]', ylabel=r'$\psi(r)$', title='Silicon pseudopotential', labels=['3s', '3p'])

    print('\n')

    # Compute a scattering state
    E = 1.0 # Hartree

    print(f'Computing scattering state at E = {E:.2f} Ha')
    vn_s = calc.GetScatt(E, l=0)
    vn_p = calc.GetScatt(E, l=1)
    vn_d = calc.GetScatt(E, l=2)

    wfn_s = calc.GetWavefunc(vn_s, rplot, bound=False, l=0)
    wfn_p = calc.GetWavefunc(vn_p, rplot, bound=False, l=1)
    wfn_d = calc.GetWavefunc(vn_d, rplot, bound=False, l=2)

    # Only pack the real part
    wfn_scatt = np.column_stack((wfn_s.real, wfn_p.real, wfn_d.real))

    Plot(rplot, wfn_scatt, xlabel='r [Bohr]', ylabel=r'Real [$\psi(r)$]', title='Silicon scattering states', labels=['s', 'p', 'd'])

    print('\n')

    # Compute the scattering phase
    # Should start from Ek > 0.0
    KinE = np.linspace(0.1, 6.0, 100)

    print('Computing scattering phase')
    smooth = True
    phase_s = calc.GetScatteringPhase(KinE, l=0, smooth=smooth)
    phase_p = calc.GetScatteringPhase(KinE, l=1, smooth=smooth)
    phase_d = calc.GetScatteringPhase(KinE, l=2, smooth=smooth)
    phase = np.column_stack((phase_s, phase_p, phase_d))

    Plot(KinE, phase, xlabel='Kinetic energy [Ha]', ylabel=r'Phase [$\pi$]', title='Silicon scattering phase', labels=['s', 'p', 'd'])

    print('\n')

    # Compute the radial integral
    smooth = True
    print('Computing radial integrals for 3s')
    radint_3s_abs, radint_3s_angle = calc.GetRadialIntegral(KinE, n=1, l=0, verbose='low', store_type='abs-angle', smooth_phase=smooth)

    print('Computing radial integrals for 3p')
    radint_3p_abs, radint_3p_angle = calc.GetRadialIntegral(KinE, n=2, l=1, verbose='low', store_type='abs-angle', smooth_phase=smooth)

    radint_abs = np.column_stack((radint_3s_abs, radint_3p_abs))
    radint_angle = np.column_stack((radint_3s_angle, radint_3p_angle))
    
    Plot(KinE, radint_abs, xlabel='Kinetic energy [Ha]', ylabel=r'Absolute value', title='Silicon radial integrals', 
         labels=[r'$s \rightarrow p$', r'$p \rightarrow s$', r'$p \rightarrow d$'])
    Plot(KinE, radint_angle, xlabel='Kinetic energy [Ha]', ylabel=r'Phase [$\pi$]', title='Silicon radial integrals phase', 
         labels=[r'$s \rightarrow p$', r'$p \rightarrow s$', r'$p \rightarrow d$'])
# ===================================================================================================================
if __name__ == '__main__':
    main()
