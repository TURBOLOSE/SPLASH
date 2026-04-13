import numpy as np
import pandas as pd
from scipy.integrate import *
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree
from matplotlib import pyplot as plt

#import plotly.express as px


def make_input_4(): #no energy as separate variable
    face_centers=pd.read_table('results/face_centers.dat', header=None, delimiter=r"\s+")
    N=len(face_centers[0])


    face_centers=np.array(face_centers)


    l=[]



    theta_face_centers=-np.arccos(face_centers[:,2]/np.linalg.norm(face_centers, axis=1))+np.pi/2


    #rho=np.ones(N)
    omega=np.array([0,0,2])

    #rho=np.ones(N)
    r=np.sqrt(face_centers[:,1]**2+face_centers[:,2]**2+face_centers[:,0]**2)
    rho=np.exp(-1/2*(np.linalg.norm(omega)**2)*np.sin(-np.arccos(face_centers[:,2]/r)+np.pi/2)**2)

    for face_num, R in enumerate(face_centers):
        #if( R[2] >0):
        #   omega=np.array([0,0,0.5])
        #elif ( R[2] <0):
        #    omega=np.array([0,0,-0.5])
        #else:
        #    omega=np.array([0,0,0])
        l.append(rho[face_num]*np.cross(R,np.cross(omega,R))/(np.linalg.norm(R)**2))
    l=np.array(l)

    pd.DataFrame(data=np.array([rho, l[:,0],l[:,1],l[:,2]]).transpose()).to_csv('input/input.dat',index=False, sep=' ', header=False)

def make_input_4_cos_bell(): #no energy as separate variable
    face_centers=pd.read_table('results/face_centers.dat', header=None, delimiter=r"\s+")
    N=len(face_centers[0])


    face_centers=np.array(face_centers)


    l=[]
    theta=-np.arccos(face_centers[:,2])+np.pi/2 #lat
    phi=np.arctan2(face_centers[:,1],face_centers[:,0]) #long

    rho=np.ones(N)
    omega=np.array([0,0,0])




    for face_num, R in enumerate(face_centers):
        if( abs(theta[face_num])<np.pi/6):
            if(abs(phi[face_num])<np.pi/6):
                rho[face_num]=1+10*np.cos(2.2*np.sqrt(theta[face_num]**2+phi[face_num]**2))
                if(rho[face_num]<0):
                    print('check input rho')
                omega=np.array([0,0,2])

        #elif ( R[2] <0):
        #    omega=np.array([0,0,-2])
        #else:
        #    omega=np.array([0,0,0])
        l.append(rho[face_num]*np.cross(R,np.cross(omega,R))/(np.linalg.norm(R)**2))
        omega=np.array([0,0,0])
    l=np.array(l)

    pd.DataFrame(data=np.array([rho, l[:,0],l[:,1],l[:,2]]).transpose()).to_csv('input/input.dat',index=False, sep=' ', header=False)



def make_input_5():
    gam0=5./3
    gam=2-1/gam0
    #face_centers=pd.read_table('results/face_centers_ico_6.dat', header=None, delimiter=r"\s+")
    face_centers=pd.read_table('results/face_centers.dat', header=None, delimiter=r"\s+")
    N=len(face_centers[0])


    face_centers=np.array(face_centers)

    rho=np.ones(N)*10
    omega=np.array([0,0,0.1])
    #p=np.ones(N)
    p=1+(np.linalg.norm(omega)**2*rho/2*np.sin(-np.arccos(face_centers[:,2]))**2)
    l=[]
    v=[]

    
    theta=-np.arccos(face_centers[:,2]/np.linalg.norm(face_centers, axis=1))+np.pi/2 #lat
    phi=np.arctan2(face_centers[:,1]/np.linalg.norm(face_centers, axis=1),face_centers[:,0]/np.linalg.norm(face_centers, axis=1)) #long
    a=1
    R0=a/3
    #r=a*np.arccos(np.sin(0)*np.sin(theta)+np.cos(theta)*np.cos(0)*np.cos(phi-np.pi*3/2))
    theta_c=0
    phi_c=0.5
    r=a*np.arccos(np.sin(theta_c)*np.sin(theta)+np.cos(theta)*np.cos(theta_c)*np.cos(phi-phi_c))


    for face_num, R in enumerate(face_centers):
        # if((theta[face_num] >-np.pi/4 and theta[face_num]<np.pi/4)):
        # #if( (theta[face_num] >0 and theta[face_num]<np.pi/4) or (theta[face_num] >-np.pi/2 and theta[face_num]<-np.pi/4) ):
        #     omega=np.array([0,0,0.5])
        # #elif ((theta[face_num] <0 and theta[face_num] >-np.pi/4) or (theta[face_num] >np.pi/4 and theta[face_num] < np.pi/2)     ):
        # elif((theta[face_num] <-np.pi/4 and theta[face_num] >-np.pi/2) or (theta[face_num] >np.pi/4 and theta[face_num] < np.pi/2)     ):
        #     omega=np.array([0,0,-0.5])
        # else:
        #     omega=np.array([0,0,0])


        # if(np.abs(theta[face_num]-np.pi/2)<0.1):
        #   p[face_num]=1
        if r[face_num]<R0:
            rho[face_num]*=1.1
        #    #p[face_num]=350

        
        # if(theta[face_num] >0):
        #     omega=np.array([0,0,0.5])
        # elif(theta[face_num]<0):
        #     omega=np.array([0,0,-0.5])
        # else:
        #     omega=np.array([0,0,0])


        l.append(rho[face_num]*np.cross(R,np.cross(omega,R))/(np.linalg.norm(R)**2))
        v.append(np.cross(omega,R)/np.linalg.norm(R))
        #print(np.cross(R,-np.cross(R, l[face_num]))/(np.linalg.norm(R)**2))

        
    l=np.array(l)
    v=np.array(v)

    print('mean_Mach= ',np.mean(np.linalg.norm(v, axis=1)/np.sqrt(gam*p/rho)))
    print('max c_s= ',np.max(gam *p/rho))

    E=1/(gam-1)*p+rho*np.linalg.norm(v, axis=1)*np.linalg.norm(v, axis=1)/2
    if(E.any()<0):
        print('Energy<0!!')

    pd.DataFrame(data=np.array([rho, l[:,0],l[:,1],l[:,2],E]).transpose()).to_csv('input/input.dat',index=False, sep=' ', header=False)

def make_input_5_coriolis():
    gam0=5./3
    gam=2-1/gam0
    #face_centers=pd.read_table('results/face_centers_ico_6.dat', header=None, delimiter=r"\s+")
    face_centers=pd.read_table('results/face_centers.dat', header=None, delimiter=r"\s+")
    N=len(face_centers[0])


    face_centers=np.array(face_centers)

    #rho_0=10
    #p_0=0.01
    rho_0=5
    p_0=1e-2
    omega0=np.array([0,0,0.1])

    #omega=np.array([0,0,0.05])
    # Set the local cyclone rotation purely in the Z axis since it's at the pole
    omega_cyclone = 0.05
    omega=np.array([0,0,omega_cyclone])

    a_0=np.sqrt(gam*p_0/rho_0)

    M_0=np.linalg.norm(omega)/a_0
    print('M_0= ', M_0)

    #M_0=(gam-1)/p_0 *np.linalg.norm(omega)**2

    rho=np.ones(N, dtype='d')*rho_0 
    p=np.ones(N, dtype='d')*p_0 
    theta=np.arccos((face_centers[:,2])/np.linalg.norm(face_centers, axis=1)) 

    #p=0.5+(np.linalg.norm(omega)**2*rho/2*np.sin(theta)**2)



    #rho=rho_0*(1+(gam-1)/2*M_0**2*np.sin(theta, dtype='d')**2)**(1/(gam-1))
    #p=p_0*(1+(gam-1)/2*M_0**2*np.sin(theta, dtype='d')**2)**(gam/(gam-1))


    l=[]
    v=[]

    
    lon=-np.arccos(face_centers[:,2]/np.linalg.norm(face_centers, axis=1))+np.pi/2 #lat
    phi=np.arctan2(face_centers[:,1]/np.linalg.norm(face_centers, axis=1),face_centers[:,0]/np.linalg.norm(face_centers, axis=1)) #long
    #a=1
    a=2
    R0=a/3
    
    #r=a*np.arccos(np.sin(0)*np.sin(lon)+np.cos(lon)*np.cos(0)*np.cos(phi-np.pi*3/2))
    #theta_c=np.pi/2
    theta_c=0#0.3
    phi_c=0
    r=a*np.arccos(np.sin(theta_c)*np.sin(lon)+np.cos(lon)*np.cos(theta_c)*np.cos(phi-phi_c))


    for face_num, R in enumerate(face_centers):

        #if r[face_num]<R0:
        #    rho[face_num]*=1.2


        #if(theta[face_num]<np.pi/2+np.arccos(1/np.sqrt(2)) and theta[face_num]>np.pi/2-np.arccos(0.3/np.sqrt(2))):

        #if(theta[face_num]<np.pi/2+np.arccos(1/np.sqrt(2)) and theta[face_num]>np.pi/2-np.arccos(1/np.sqrt(2))):
        #if(theta[face_num]<np.arccos(1/np.sqrt(2))):
        # if(r[face_num]<R0):
        #     # Smoothly taper the velocity to zero using a cosine bell
        #     smooth_factor = 0.5 * (1 - np.cos(np.pi * r[face_num] / R0))
        #     local_omega = np.array([omega_cyclone * smooth_factor,0,0])

        #     l.append(rho[face_num]*np.cross(R,np.cross(local_omega,R))/(np.linalg.norm(R)**2))
        #     v.append(np.cross(local_omega,R)/np.linalg.norm(R))
            
        #     rho[face_num]*=1.2
        #     dp = 0.5 * rho[face_num] * (np.linalg.norm(v[face_num])**2) * (1 - np.cos(np.pi * r[face_num] / R0))
        #     p[face_num] += dp

        # else:
        #     l.append(np.array([0,0,0]))
        #     v.append(np.array([0,0,0]))

        # if(theta[face_num]<np.pi/2+0.5+0.3 and theta[face_num]>np.pi/2+0.5-0.3):
        #     l.append(rho[face_num]*np.cross(R,np.cross(omega,R))/(np.linalg.norm(R)**2))
        #     v.append(np.cross(omega,R)/np.linalg.norm(R))
        # else:
        #     l.append(np.array([0,0,0]))
        #     v.append(np.array([0,0,0]))
        l.append(np.array([0,0,0]))
        v.append(np.array([0,0,0]))

    #p=1+(np.linalg.norm(omega)**2*rho/2*np.sin(theta)**2)

    l=np.array(l) 
    v=np.array(v) 

    theta_l = np.linspace(0,np.pi,200)

    # sort theta and rho for safe interpolation
    theta_sorted_idx = np.argsort(theta)
    theta_sorted = theta[theta_sorted_idx]
    rho_sorted = rho[theta_sorted_idx]
    vel_sorted=np.linalg.norm(v[theta_sorted_idx], axis=1)


    # define pressure ODE dp/dtheta = ||omega||^2/2 * sin(theta)*cos(theta) * rho(theta)
    def press(p, th):
        # We need np.tan(th) for the curvature term on a sphere, not np.tanh
        tan_th = np.tan(th)
        if np.abs(tan_th) < 1e-5:
            cot_th = 0
        else:
            cot_th = 1 / tan_th
            
        return  np.interp(th, theta_sorted, rho_sorted) * (
            np.interp(th, theta_sorted, vel_sorted) ** 2 * cot_th
            + 2 * np.linalg.norm(omega0) * np.interp(th, theta_sorted, vel_sorted) * np.sin(th)
        )

    # hmax ensures the ODE solver doesn't skip over the band where velocity is non-zero
    #sol = odeint(press, p_0, theta_l, hmax=0.01)

    # if sol.ndim == 1:
    #     p_short = sol
    # else:
    #     p_short = sol[:, 0]

    # p=np.interp(theta, theta_l, p_short)
    # p+=rho*np.linalg.norm(omega0)**2  * np.sin(theta)**2/2


    #p=p_0*(rho/rho_0)**gam



    #print(p_short)
    #print(p)

   #v=np.array(v)

    print('mean_Mach= ',np.mean(np.linalg.norm(v, axis=1)/np.sqrt(gam*p/rho)))
    print('max c_s= ',np.max(gam *p/rho))
    if(np.max(gam *p/rho)>1/np.sqrt(3)):
        print('Warning: c_s is too high!')

    #E=1/(gam-1)*p+rho*np.linalg.norm(v, axis=1)*np.linalg.norm(v, axis=1)/2+rho*np.linalg.norm(omega)**2*(np.sin(theta)**2)/2
    E=1/(gam-1)*p+rho*np.linalg.norm(v, axis=1)*np.linalg.norm(v, axis=1)/2
    if(E.any()<0):
        print('Energy<0!!')

    pd.DataFrame(data=np.array([rho, l[:,0],l[:,1],l[:,2],E]).transpose()).to_csv('input/input.dat',index=False, sep=' ', header=False)

def make_input_geostrophic_pressure_first():
    """
    SPHEREPACK-style initialization:
    1. Prescribe pressure perturbation as a Gaussian blob
    2. Compute its gradient on the icosahedral mesh (LSQ reconstruction)
    3. Invert geostrophic balance to get velocity:  f * r̂ × v = -∇Π/Σ
       => v = (r̂ × ∇Π) / (Σ * f)
    4. Compute angular momentum l from v
    5. Energy from v and Π
    """

    gam0 = 5./3
    gam  = 2 - 1/gam0      # shallow water effective gamma

    face_centers = np.array(pd.read_table('results/face_centers.dat',
                            header=None, delimiter=r"\s+"))
    
    with open('results/neighbors.dat', 'r') as f:
        neighbors = [[int(x) for x in line.split()] for line in f]

    face_centers=face_centers/np.array([np.linalg.norm(face_centers,axis=1),np.linalg.norm(face_centers,axis=1),np.linalg.norm(face_centers,axis=1)]).T
    N = len(face_centers)

    # ----------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------
    rho_0        = 1
    #p_0          = 4e-4
    Omega_frame  = 0.1
    omega0       = np.array([0.0, 0.0, Omega_frame])

    lat_c=0.7

    cos_tc      = np.sin(lat_c)  
    f_0    = 2 * Omega_frame * cos_tc
    # Vortex centre (arbitrary latitude/longitude)
    theta_c = np.pi/2-lat_c          # colatitude from north pole
    phi_c   = 0.5          # longitude

   # Ro_target  = 0.05     # Rossby number
   # Ma_target  = 1
    sigma_r  = 0.1        # Gaussian width in radians
   # c_s_target    = f_0 * sigma_r / (Ma_target / Ro_target)
    #p_0 = c_s_target ** 2 * rho_0
    p_0=1e-4

    # Pressure perturbation amplitude and width
    # Negative = low pressure = cyclone (Northern Hemisphere)
    delta_Pi = -0.05 * p_0
    #delta_Pi = -0.05 * p_0
    #delta_Pi = 0.05 *p_0

    # ----------------------------------------------------------------
    # Geometry
    # ----------------------------------------------------------------
    R_norm   = np.linalg.norm(face_centers, axis=1)
    R_sphere = np.mean(R_norm)
    n_hat    = face_centers / R_norm[:, None]   # unit normals

    # Coriolis parameter f = 2Ω·r̂  (= 2Ω cosθ)
    # n_hat[:,2] = cosθ
    f_cor = 2.0 * Omega_frame * n_hat[:, 2]     # shape (N,)

    # Unit normal at vortex centre
    n_c = np.array([
        np.sin(theta_c) * np.cos(phi_c),
        np.sin(theta_c) * np.sin(phi_c),
        np.cos(theta_c)
    ])

    # Great-circle distances from vortex centre
    cos_dist = np.clip(n_hat @ n_c, -1.0, 1.0)
    r_gc     = R_sphere * np.arccos(cos_dist)   # GCD

    # ----------------------------------------------------------------
    # Step 1: Prescribe pressure field
    # Gaussian in great-circle angle (dimensionless, angle in radians)
    # ----------------------------------------------------------------
    angle_gc = r_gc / R_sphere   # great-circle angle, radians

    p = p_0 + delta_Pi * np.exp(-angle_gc**2 / (2.0 * sigma_r**2))

    # Density uniform (no centrifugal, Coriolis does not compress)
    rho = np.ones(N) * rho_0

    # ----------------------------------------------------------------
    # Step 2: Compute ∇Π on the mesh
    # Use weighted least-squares gradient reconstruction:
    # For each cell i with neighbours j:
    #   Π_j - Π_i ≈ ∇Π|_i · (r_j - r_i)_tangential
    # Solved in the local tangent plane of cell i.
    # ----------------------------------------------------------------

    # Find neighbours via KDTree (include enough to overdetermine the 2D system)

    #tree    = KDTree(face_centers)
    #N_neigh = 9    # number of neighbours (excluding self); 9 > 2 DOF, robust
    #dists, idx = tree.query(face_centers, k=N_neigh + 1)
    #print(dists)


    dists=[]
    for i,face in enumerate(face_centers):
        temp=[]
        for n in neighbors[i]:
            temp.append(2*np.arcsin(np.linalg.norm(face_centers[n] - face_centers[i])/2))
        dists.append(temp)

    idx=neighbors

    # idx[i] contains neighbours

    grad_Pi = np.zeros((N, 3))   # 3D gradient (will be projected to tangent plane)

    # for i in range(N):
    #     ni  = n_hat[i]          # unit normal at cell i
    #     Pi_i = p[i]

    #     # Build two orthogonal tangent vectors at cell i
    #     # e1: arbitrary vector not parallel to ni
    #     ref = np.array([1.0, 0.0, 0.0])
    #     if abs(ni @ ref) > 0.9:
    #         ref = np.array([0.0, 1.0, 0.0])
    #     e1 = np.cross(ni, ref);  e1 /= np.linalg.norm(e1)
    #     e2 = np.cross(ni, e1)    # already unit since ni⊥e1

    #     # Accumulate least-squares system A x = b, x = [∂Π/∂e1, ∂Π/∂e2]
    #     A = []
    #     b = []
    #     for j in idx[i]:    # skip self no longer needed since neighbors.dat only has neighbors
    #         dr    = face_centers[j] - face_centers[i]
    #         # Project dr onto tangent plane of cell i
    #         dr_t  = dr - (dr @ ni) * ni
    #         A.append([dr_t @ e1, dr_t @ e2])
    #         b.append(p[j] - Pi_i)

    #     A = np.array(A)
    #     b = np.array(b)

    #     # Weighted by inverse distance (closer neighbours count more)
    #     w = 1.0 / (np.linalg.norm(
    #             face_centers[idx[i]] - face_centers[i], axis=1) + 1e-12)
    #     W = np.diag(w)

    #     # Solve weighted LSQ: (AᵀWA) x = AᵀWb
    #     AtW  = A.T @ W
    #     coef, _, _, _ = np.linalg.lstsq(AtW @ A, AtW @ b, rcond=None)

    #     # Back to 3D (already tangential to sphere)
    #     grad_Pi[i] = coef[0] * e1 + coef[1] * e2


    for i in range(N):
        alpha_i   = angle_gc[i]           # great-circle angle, radians
        sin_alpha = np.sqrt(1.0 - cos_dist[i]**2)

        # Skip the vortex centre and antipode where sin(alpha)→0
        if sin_alpha < 1e-10:
            continue
        tangent = n_c - cos_dist[i] * n_hat[i] 

        grad_Pi[i] = (
            (alpha_i / sigma_r**2)
            * delta_Pi
            * np.exp(-alpha_i**2 / (2.0 * sigma_r**2))
            * tangent / sin_alpha
        )   

    print(np.insert(grad_Pi[:,0],0,0))
    pd.DataFrame(np.insert(grad_Pi[:,0],0,0)).to_csv('results/gradp.dat', index=False, sep=' ', header=False)


    # ----------------------------------------------------------------
    # Step 3: Geostrophic velocity reconstruction
    # Geostrophic balance:  Σ f (r̂ × v) = -∇Π
    # Since r̂ × v is tangential and |r̂| = 1:
    #   r̂ × v = -∇Π / (Σ f)
    # Take cross product with r̂ from left:
    #   r̂ × (r̂ × v) = -r̂ × ∇Π / (Σ f)
    #   -v = -r̂ × ∇Π / (Σ f)        [since r̂×(r̂×v) = -v for tangential v]
    #   v  =  r̂ × ∇Π / (Σ f)
    # ----------------------------------------------------------------

    v = np.zeros((N, 3))

    # Mask cells too close to equator where f→0 (geostrophic balance breaks)


    f_min   = 2.0 * Omega_frame * np.cos(np.radians(80))  # within 10° of equator
    eq_mask = np.abs(f_cor) > f_min

    # # Tropics: set v=0 (no geostrophic balance possible)
    # v[~eq_mask] = 0.0

    v[eq_mask] = (
        np.cross(n_hat[eq_mask], grad_Pi[eq_mask])
        / (rho[eq_mask, None] * f_cor[eq_mask, None])
    )

    # ----------------------------------------------------------------
    # Step 4: Enforce r̂·v = 0 exactly (remove any radial leakage)
    # ----------------------------------------------------------------
    vdotn = np.einsum('ij,ij->i', v, n_hat)
    v    -= vdotn[:, None] * n_hat

    # ----------------------------------------------------------------
    # Step 5: Angular momentum  l = Σ (r × v) / |r|
    # l_i = rho * cross(R_vec, v) / |R|   [units: surface density × velocity]
    # ----------------------------------------------------------------
    l = np.zeros((N, 3))
    for i in range(N):
        l[i] = rho[i] * np.cross(face_centers[i], v[i]) / R_norm[i]

    # ----------------------------------------------------------------
    # Step 6: Energy
    # ----------------------------------------------------------------
    E = 1.0/(gam - 1.0) * p + rho * np.sum(v**2, axis=1) / 2.0

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------
    c_s  = np.sqrt(gam * p / rho)
    mach = np.linalg.norm(v, axis=1) / c_s
    rv   = np.max(np.abs(np.einsum('ij,ij->i', n_hat, v)))

    print(f'Pressure min/max         : {np.min(p):.6f} / {np.max(p):.6f}')
    print(f'ΔΠ (centre)              : {np.min(p) - p_0:.6f}  (expected {delta_Pi:.6f})')
    print(f'Max |v|                  : {np.max(np.linalg.norm(v, axis=1))}')
    print(f'Max Mach                 : {np.max(mach)}')
    print(f'Max c_s                 : {np.max(gam*p/rho)}')
    print(f'Max |r̂·v| (tangency)    : {rv:.2e} ')
    #print(f'Cells zeroed (equator)   : {np.sum(~eq_mask)}')

    # Estimated geostrophic velocity scale:  V ~ ΔΠ / (Σ f R σ)
    f_c    = 2.0 * Omega_frame * np.cos(theta_c)
    V_est  = abs(delta_Pi) / (rho_0 * f_c * R_sphere * sigma_r)
    Ro_est = V_est / (f_c * R_sphere * sigma_r)
    print(f'Estimated velocity scale : {V_est:.4f}')
    print(f'Cyclone radius           : {sigma_r:.4f}')
    print(f'Estimated Rossby radius  : {np.sqrt((2*gam-1)/(gam-1)*p_0/rho_0)/(2*np.linalg.norm(Omega_frame)):.4f}')
    print(f'Estimated Rossby number  : {Ro_est:.4f}  (<<1 needed for geostrophic validity)')

    pd.DataFrame(
        np.column_stack([rho, l[:, 0], l[:, 1], l[:, 2], E])
    ).to_csv('input/input.dat', index=False, sep=' ', header=False)


def make_input_geostrophic_coriolis():
    gam0 = 5./3
    gam  = 2 - 1/gam0

    face_centers = np.array(pd.read_table('results/face_centers.dat',
                            header=None, delimiter=r"\s+"))

    face_centers=face_centers/np.array([np.linalg.norm(face_centers,axis=1),np.linalg.norm(face_centers,axis=1),np.linalg.norm(face_centers,axis=1)]).T
    N = len(face_centers)

    rho_0        = 1.0
    p_0          = 1e-6
    Omega_frame  = 0.1
    omega_cyclone = 0.005

    # Vortex center (arbitrary, non-polar)
    theta_c = np.pi/2-0.5
    phi_c   = 0.1 #0.5

    # Unit normal at vortex center = rotation axis for the vortex
    n_c = np.array([
        np.sin(theta_c) * np.cos(phi_c),
        np.sin(theta_c) * np.sin(phi_c),
        np.cos(theta_c)
    ])

    R_norm   = np.linalg.norm(face_centers, axis=1)
    n_hat    = face_centers / R_norm[:, None]   # unit vectors to all cells
    R_sphere=1

    R0 = R_sphere / 3.0                  # vortex radius

    # --- Great-circle distances from vortex center ---
    cos_dist = np.clip(n_hat @ n_c, -1.0, 1.0)
    r_gc     = R_sphere * np.arccos(cos_dist)   # shape (N,)

    # --- Analytic cosine-bell tangential speed ---
    # v(r) = omega_local(r) * r,  omega_local = omega_cyclone * bell(r)
    def v_analytic(r_arr):
        r_arr = np.asarray(r_arr, dtype=float)
        v_out = np.zeros_like(r_arr)
        mask  = r_arr < R0
        v_out[mask] = (omega_cyclone
                       * 0.5 * (1 - np.cos(np.pi * r_arr[mask] / R0))
                       * r_arr[mask])
        return v_out

    # --- Velocity vectors and angular momentum ---
    rho = np.ones(N) * rho_0
    v   = np.zeros((N, 3))
    l   = np.zeros((N, 3))

    for i, R_vec in enumerate(face_centers):
        if r_gc[i] < R0:
            smooth      = 0.5 * (1 - np.cos(np.pi * r_gc[i] / R0))
            local_omega = omega_cyclone * smooth * n_c    # axis = n_c always
            v[i] = np.cross(local_omega, R_vec) / R_norm[i]
            l[i] = (rho[i]
                    * np.cross(R_vec, np.cross(local_omega, R_vec))
                    / R_norm[i]**2)
        else:
            v[i]=np.zeros(3)
            l[i]=np.zeros(3)

    # --- 2D geostrophic pressure integration ---
    #
    # For each cell i, integrate along the great-circle from vortex center to cell i:
    #
    #   P(s) = n_c * cos(s) + t_hat_i * sin(s),   s in [0, r_gc[i]/R]
    #
    # cos(theta) along path = P_z(s) = n_c[2]*cos(s) + t_hat_i[2]*sin(s)
    #
    # This is DIFFERENT for every cell i because t_hat_i points in a different
    # direction — this is why a single 1D ODE over r was wrong.
    #
    # dPi/dr = rho_0 * 2*Omega * cos(theta(r)) * v_analytic(r)

    N_quad = 10000    # quadrature points per path — enough to resolve the bell
    p = np.ones(N) * p_0

    # Precompute quadrature nodes on [0,1] (rescaled per cell)
    s_nodes = np.linspace(0.0, 1.0, N_quad)

    for i in range(N):
        dot = cos_dist[i]   # n_c . n_hat[i], already computed
        # Skip cells at the vortex center or its antipode
        if np.abs(dot) > 1.0 - 1e-10:
            continue

        # Unit great-circle tangent at n_c pointing toward cell i
        perp  = n_hat[i] - dot * n_c
        t_hat = perp / np.linalg.norm(perp)   # lies in tangent plane of n_c
        # Arc angle to cell (= r_gc[i] / R_sphere)
        s_max = np.arccos(np.clip(dot, -1.0, 1.0))  # in radians
        # Quadrature points along the great-circle arc
        s   = s_nodes * s_max                        # shape (N_quad,)
        r_s = s * R_sphere                           # actual distances

        # cos(theta) at each quadrature point — KEY: uses t_hat specific to cell i
        cos_theta_s = n_c[2] * np.cos(s) + t_hat[2] * np.sin(s)

        # Coriolis parameter and speed at each quadrature point
        f_s   = 2.0 * Omega_frame * cos_theta_s
        v_s   = v_analytic(r_s)

        # Integrate dPi/dr = rho * f * v  using trapezoid rule
        integrand = rho_0 * f_s * v_s
        p[i] = p_0 + np.trapezoid(integrand, r_s)#*(gam-1)/(2*gam-1)

        outside = r_gc > R0 * 1.1
        p[outside] = p_0  
        #p[i]*=

    # --- Diagnostics ---
    c_s  = np.sqrt(gam * p / rho)
    mach = np.linalg.norm(v, axis=1) / c_s
    print(f'Max Mach          = {np.max(mach):.4f}')
    print(f'Pressure min/max  = {np.min(p):.4f} / {np.max(p):.4f}')
    print(f'ΔΠ (centre→edge)  = {np.max(p) - np.min(p):.4f}')
    print(f'Expected ΔΠ ≈ Σ·2Ω·ω₁·R₀²·cos(θ_c) = '
          f'{rho_0 * 2*Omega_frame * omega_cyclone * R0**2 * np.cos(theta_c):.4f}')

    rv_max = np.max(np.abs(np.einsum('ij,ij->i', n_hat, v)))
    print(f'Max |r̂·v|         = {rv_max:.2e}  (should be ~machine eps)')

    # --- Energy and output ---
    E = 1/(gam - 1) * p + rho * np.sum(v**2, axis=1) / 2



    pd.DataFrame(
        np.column_stack([rho, l[:, 0], l[:, 1], l[:, 2], E])
    ).to_csv('input/input.dat', index=False, sep=' ', header=False)

def make_input_geostrophic_pressure_first2():
    gam0 = 5./3
    gam  = 2 - 1/gam0

    face_centers = np.array(pd.read_table('results/face_centers.dat',
                            header=None, delimiter=r"\s+"))
    face_centers = face_centers / np.linalg.norm(
                        face_centers, axis=1, keepdims=True)

    theta=np.arccos(face_centers[:,2]) 

    N = len(face_centers)
    with open('results/neighbors.dat', 'r') as f:
        neighbors = [[int(x) for x in line.split()] for line in f]

    # ----------------------------------------------------------------
    # Parameters: choose p_0 so that sigma_r/L_R ~ 2-3 and Ma ~ 0.1
    # ----------------------------------------------------------------
    Omega_frame = 0.1
    lat_c       = 0.7
    theta_c     = np.pi/2 - lat_c
    phi_c       = 0.5
    sigma_r     = 0.1

    cos_tc = np.sin(lat_c)
    f_0    = 2.0 * Omega_frame * cos_tc

    # Target Ma/Ro = sigma_r/L_R = 2  =>  L_R = sigma_r/2
    # c_s = f_0 * L_R = f_0 * sigma_r / 2
    #c_s_target = f_0 * sigma_r / 2.0      # gives sigma_r/L_R = 2

    c_s_target = 6e-5
    rho_0 = 1.0
    p_0=10e-06

   #p_0   = rho_0 * c_s_target**2 / gam   # from c_s^2 = gam*p/rho
    # EOS constant: Pi = K * Sigma^gam
    K     = p_0 / rho_0**gam

    # Small perturbation: Ro ~ 0.1
    Ro_target = 0.03
    V_target  = Ro_target * f_0 * sigma_r
    # Geostrophic scale: delta_Pi ~ rho_0 * f_0 * V * sigma_r
    delta_Pi  = - rho_0 * f_0 * V_target * sigma_r


    print(f'p_0          = {p_0:.6e}')
    print(f'c_s          = {c_s_target:.6f}')
    print(f'L_R          = {c_s_target/f_0:.6f}  ({c_s_target/f_0/sigma_r:.2f} sigma_r)')
    print(f'delta_Pi/p_0 = {delta_Pi/p_0*100:.3f}%')

    # ----------------------------------------------------------------
    # Geometry
    # ----------------------------------------------------------------
    R_norm   = np.linalg.norm(face_centers, axis=1)
    R_sphere = np.mean(R_norm)
    n_hat    = face_centers / R_norm[:, None]
    f_cor    = 2.0 * Omega_frame * n_hat[:, 2]

    n_c = np.array([np.sin(theta_c)*np.cos(phi_c),
                    np.sin(theta_c)*np.sin(phi_c),
                    np.cos(theta_c)])

    cos_dist = np.clip(n_hat @ n_c, -1.0, 1.0)
    angle_gc = np.arccos(cos_dist)
    bell     = np.exp(-angle_gc**2 / (2.0 * sigma_r**2))

    # ----------------------------------------------------------------
    # Step 1: Pressure field
    # ----------------------------------------------------------------
    p = p_0 + delta_Pi * bell

    # ----------------------------------------------------------------
    # Step 2: CONSISTENT density from EOS  Pi = K Sigma^gam
    # This is the key fix vs. uniform rho
    # ----------------------------------------------------------------
    rho = (p / K) ** (1.0 / gam)
    #rho=np.ones(N)*rho_0

    # ----------------------------------------------------------------
    # Step 3: Analytical gradient of Pi
    # ----------------------------------------------------------------
    grad_Pi = np.zeros((N, 3))
    for i in range(N):
        sin_a = np.sqrt(1.0 - cos_dist[i]**2)
        if sin_a < 1e-10:
            continue
        tangent    = n_c - cos_dist[i] * n_hat[i]
        grad_Pi[i] = (angle_gc[i] / sigma_r**2 * delta_Pi * bell[i]
                      * tangent / sin_a)

    # Sanity: gradient must be tangential
    leak = np.max(np.abs(np.einsum('ij,ij->i', n_hat, grad_Pi)))
    print(f'Max |r̂·∇Π|  = {leak:.2e}  (EOS-consistent gradient)')

    # ----------------------------------------------------------------
    # Step 4: Geostrophic velocity  v = (r̂ × ∇Π) / (Σ f)
    # Now uses the ACTUAL rho(r), not rho_0
    # ----------------------------------------------------------------
    v     = np.zeros((N, 3))
    f_min = 2.0 * Omega_frame * np.cos(np.radians(88))
    mask  = np.abs(f_cor) > f_min

    v[mask] = (np.cross(n_hat[mask], grad_Pi[mask])/ (rho[mask, None] * f_cor[mask, None]))
    #plt.scatter(theta[mask], np.linalg.norm(v[mask], axis=1))
    #plt.savefig('plots/wtf.png', bbox_inches='tight',dpi=300)

    #print(np.linalg.norm(v[mask], axis=1))
    #v[mask] = 0

    #v = (np.cross(n_hat, grad_Pi) / (rho[:, None] * f_cor[:, None]))

    # Enforce tangency
    #v -= np.einsum('ij,ij->i', v, n_hat)[:, None] * n_hat
    #v = make_divergence_free(v, rho, face_centers, neighbors, n_hat, R_sphere)

    # ----------------------------------------------------------------
    # Step 5: Angular momentum and energy
    # ----------------------------------------------------------------
    l = rho[:, None] * np.cross(face_centers, v)  # R_sphere=1 so /|R|=1

    E = 1.0/(gam - 1.0) * p + rho * np.sum(v**2, axis=1) / 2.0

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------
    c_s  = np.sqrt(gam * p / rho)
    mach = np.linalg.norm(v, axis=1) / c_s
    rho_var = (np.max(rho) - np.min(rho)) / rho_0

    print(f'rho variation            : {rho_var*100:.4f}%  (EOS consistency)')
    print(f'Max Mach                 : {np.max(mach):.4f}')
    print(f'Max |v|                  : {np.max(np.linalg.norm(v,axis=1)):.6f}')
    print(f'sigma_r / L_R            : {sigma_r * f_0 / c_s_target:.3f}')
    print(f'Rossby number            : {np.max(np.linalg.norm(v,axis=1))/(f_0*sigma_r):.4f}')
    print(f'EOS residual max         : '
          f'{np.max(np.abs(p - K*rho**gam)):.2e}  (should be ~0)')

    pd.DataFrame(
        np.column_stack([rho, l[:, 0], l[:, 1], l[:, 2], E])
    ).to_csv('input/input.dat', index=False, sep=' ', header=False)


from scipy.sparse import lil_matrix
from scipy.sparse.linalg import minres

def make_divergence_free(v_g, rho, face_centers, neighbors, n_hat, R_sphere):
    """
    Project v_g onto the divergence-free subspace by solving:
        ∇²ψ = -∇·(Σ v_g) / Σ
    then correcting:
        v = v_g + r̂ × (r̂ × ∇ψ)  [= v_g + ∇_s ψ]
    
    Uses finite-difference Laplacian on the icosahedral dual mesh.
    """
    N = len(face_centers)

    # ----------------------------------------------------------------
    # Step 1: Compute divergence of Sigma*v_g at each cell
    # using the face-centered finite volume divergence:
    # div(Sigma*v)|_i = (1/A_i) sum_f (Sigma_f * v_f · n_f * |e_f|)
    # Approximate with midpoint rule on each edge
    # ----------------------------------------------------------------
    div_Sv = np.zeros(N)

    for i in range(N):
        ni      = n_hat[i]
        area_i  = 4 * np.pi / N    # approximate equal-area cells

        for j in neighbors[i]:
            # Edge midpoint normal (outward from cell i)
            edge_mid = face_centers[j] - face_centers[i]
            edge_len = np.linalg.norm(edge_mid) * R_sphere

            # Outward unit normal at edge
            n_edge = edge_mid / np.linalg.norm(edge_mid)

            # Average Sigma and v at the edge
            Sig_edge = 0.5 * (rho[i] + rho[j])
            v_edge   = 0.5 * (v_g[i] + v_g[j])

            # Project velocity onto edge normal (tangential to sphere)
            # Remove radial component first
            v_edge_t = v_edge - (v_edge @ ni) * ni
            n_edge_t = n_edge - (n_edge @ ni) * ni
            n_edge_t_norm = np.linalg.norm(n_edge_t)
            if n_edge_t_norm > 1e-12:
                n_edge_t /= n_edge_t_norm

            flux = Sig_edge * (v_edge_t @ n_edge_t) * edge_len
            div_Sv[i] += flux

        div_Sv[i] /= area_i

    rhs = -div_Sv / rho    # = -∇·(Σv)/Σ

    print(f'Max |∇·(Σv)| before fix : {np.max(np.abs(div_Sv)):.3e}')
    print(f'Mean |∇·(Σv)|            : {np.mean(np.abs(div_Sv)):.3e}')

    # ----------------------------------------------------------------
    # Step 2: Build sparse graph Laplacian for ∇²ψ = rhs
    # ∇²ψ|_i ≈ (1/A_i) sum_j (ψ_j - ψ_i) / d_ij * l_ij
    # ----------------------------------------------------------------
    A_mat = lil_matrix((N, N))
    area  = 4 * np.pi / N

    for i in range(N):
        for j in neighbors[i]:
            d_ij = np.arccos(np.clip(n_hat[i] @ n_hat[j], -1, 1)) * R_sphere
            l_ij = d_ij    # approximate edge length ~ cell spacing
            w    = l_ij / (d_ij * area)
            A_mat[i, j] += w
            A_mat[i, i] -= w

    # Pin mean to zero (Neumann BC: solution defined up to constant)
    A_mat[0, :] = 0
    A_mat[0, 0] = 1
    rhs[0]      = 0

    A_csr = A_mat.tocsr()
    psi, info = minres(A_csr, rhs, rtol=1e-10, maxiter=10000)
    if info != 0:
        print(f'Warning: minres did not converge, info={info}')
    print(f'‖∇²ψ - rhs‖ residual     : {np.max(np.abs(A_csr @ psi - rhs)):.3e}')

    # ----------------------------------------------------------------
    # Step 3: Compute ∇ψ on the sphere and add to v_g
    # ----------------------------------------------------------------
    grad_psi = np.zeros((N, 3))

    for i in range(N):
        ni  = n_hat[i]
        ref = np.array([1., 0., 0.]) if abs(ni @ [1,0,0]) < 0.9 \
              else np.array([0., 1., 0.])
        e1  = np.cross(ni, ref);  e1 /= np.linalg.norm(e1)
        e2  = np.cross(ni, e1)

        A, b = [], []
        for j in neighbors[i]:
            dr   = face_centers[j] - face_centers[i]
            dr_t = dr - (dr @ ni) * ni
            A.append([dr_t @ e1, dr_t @ e2])
            b.append(psi[j] - psi[i])

        A, b = np.array(A), np.array(b)
        w    = 1.0 / (np.linalg.norm(
                    face_centers[neighbors[i]] - face_centers[i], axis=1) + 1e-12)
        W    = np.diag(w)
        AtW  = A.T @ W
        coef, _, _, _ = np.linalg.lstsq(AtW @ A, AtW @ b, rcond=None)
        grad_psi[i]   = coef[0] * e1 + coef[1] * e2

    v_corrected = v_g + grad_psi

    # Re-enforce tangency
    v_corrected -= np.einsum('ij,ij->i', v_corrected, n_hat)[:, None] * n_hat

    # Verify
    div_after = np.zeros(N)
    for i in range(N):
        area_i = 4 * np.pi / N
        for j in neighbors[i]:
            edge_mid = face_centers[j] - face_centers[i]
            edge_len = np.linalg.norm(edge_mid) * R_sphere
            n_edge   = edge_mid / np.linalg.norm(edge_mid)
            ni       = n_hat[i]
            Sig_edge = 0.5 * (rho[i] + rho[j])
            v_edge   = 0.5 * (v_corrected[i] + v_corrected[j])
            v_edge_t = v_edge - (v_edge @ ni) * ni
            n_edge_t = n_edge - (n_edge @ ni) * ni
            n_norm   = np.linalg.norm(n_edge_t)
            if n_norm > 1e-12:
                n_edge_t /= n_norm
            div_after[i] += Sig_edge * (v_edge_t @ n_edge_t) * edge_len
        div_after[i] /= area_i

    print(f'Max |∇·(Σv)| after fix  : {np.max(np.abs(div_after)):.3e}')
    return v_corrected

def make_input_rossby_haurwitz(n_wave=4, omega_rh_frac=0.1, K_frac=0.1):
    """
    Rossby-Haurwitz wave initial conditions for compressible shallow water.

    Velocity field (exact for barotropic vorticity equation):
        u_φ = ω cos(lat) + K cos^(n-1)(lat)[n sin²lat - cos²lat] cos(nφ)
        u_lat = -nK cos^(n-1)(lat) sin(lat) sin(nφ)

    Absolute vorticity (analytic):
        η = 2(Ω+ω) sin(lat) - K(n+1)(n+2) cos^n(lat) sin(lat) cos(nφ)

    Balance (Bernoulli equation):
        ∇B = -(f+ζ)(r̂×v),  B = v²/2 + γ/(γ-1) Π/Σ

    Solved as Poisson equation ∇²B = ∇·F on the icosahedral mesh.
    Density from EOS: Σ = ((γ-1)/(γ K_eos) h_ent)^(1/(γ-1))

    Expected behavior:
        The n-fold wave pattern should propagate eastward at the RH phase
        speed.  Unlike the geostrophic vortex, this is a PROPAGATING wave
        — the diagnostic is phase speed, not stationarity.

    Phase speed (angular, in inertial frame):
        ω_phase = ω_rh - 2(Ω+ω_rh) / (n(n+1)/2 - 1) × ...
        (tracked via longitude of pressure maximum)
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import minres

    gam0 = 5./3
    gam  = 2 - 1/gam0        # shallow water effective gamma

    face_centers = np.array(pd.read_table('results/face_centers.dat',
                            header=None, delimiter=r"\s+"))
    with open('results/neighbors.dat', 'r') as f_nb:
        neighbors = [[int(x) for x in line.split()] for line in f_nb]

    face_centers = face_centers / np.linalg.norm(face_centers, axis=1,
                                                  keepdims=True)
    N = len(face_centers)

    # ----------------------------------------------------------------
    # Parameters
    # Choose omega_rh/Omega and K/Omega small so that:
    #   - Rossby number Ro = V / (f * L) << 1
    #   - Mach number Ma = V / c_s ~ 0.1  (HLLC works well)
    # The Williamson standard uses omega_rh/Omega ~ 0.1, K/Omega ~ 0.1
    # ----------------------------------------------------------------
    Omega_frame  = 0.1
    rho_0        = 1.0
    n            = n_wave

    omega_rh = omega_rh_frac * Omega_frame   # background zonal flow rate
    K        = K_frac        * Omega_frame   # wave amplitude

    # Peak velocity scale: V_max ~ (|omega_rh| + |K|) * R, R=1
    V_max = (abs(omega_rh) + abs(K))
    # Set p_0 so that Ma = V_max / c_s = 0.1
    c_s_target = V_max / 0.1
    p_0        = rho_0 * c_s_target**2 / gam
    K_eos      = p_0 / rho_0**gam       # EOS constant: Pi = K_eos * Sigma^gam

    print(f'=== Rossby-Haurwitz wave  n={n} ===')
    print(f'omega_rh     = {omega_rh:.5f}   ({omega_rh_frac:.2f} * Omega)')
    print(f'K            = {K:.5f}   ({K_frac:.2f} * Omega)')
    print(f'c_s_target   = {c_s_target:.5f}')
    print(f'p_0          = {p_0:.4e}')
    print(f'Expected Ma  = {V_max/c_s_target:.3f}')

    # ----------------------------------------------------------------
    # Geometry
    # ----------------------------------------------------------------
    R_norm   = np.linalg.norm(face_centers, axis=1)
    R_sphere = np.mean(R_norm)
    n_hat    = face_centers / R_norm[:, None]

    sin_lat = n_hat[:, 2]                              # = sin(geographic lat)
    cos_lat = np.sqrt(np.maximum(1.0 - sin_lat**2, 0.0))
    phi     = np.arctan2(n_hat[:, 1], n_hat[:, 0])    # longitude

    # Orthonormal tangent basis at each cell
    # e_phi: eastward,   e_lat: northward
    e_phi = np.column_stack([-np.sin(phi),
                              np.cos(phi),
                              np.zeros(N)])

    e_lat = np.column_stack([-sin_lat * np.cos(phi),
                              -sin_lat * np.sin(phi),
                               cos_lat])

    # ----------------------------------------------------------------
    # Step 1: RH velocity field (analytic)
    #
    # u_phi = ω cosθ + K cos^(n-1)θ [n sin²θ - cos²θ] cos(nφ)
    # u_lat = -nK cos^(n-1)θ sinθ sin(nφ)
    # ----------------------------------------------------------------
    cos_n1 = cos_lat ** (n - 1)
    cos_n  = cos_lat ** n

    u_phi = (omega_rh * cos_lat
             + K * cos_n1 * (n * sin_lat**2 - cos_lat**2) * np.cos(n * phi))

    u_lat = -n * K * cos_n1 * sin_lat * np.sin(n * phi)

    # 3D velocity
    v = u_phi[:, None] * e_phi + u_lat[:, None] * e_lat

    # Enforce r̂·v = 0 exactly
    v -= np.einsum('ij,ij->i', v, n_hat)[:, None] * n_hat
    v_sq = np.sum(v**2, axis=1)

    # ----------------------------------------------------------------
    # Step 2: Analytic absolute vorticity
    #
    # ζ = (1/cosθ)[∂v/∂φ - ∂(u cosθ)/∂θ]
    #   = 2ω sinθ - K(n+1)(n+2) cos^n(θ) sinθ cos(nφ)
    #
    # η = f + ζ = 2(Ω+ω) sinθ - K(n+1)(n+2) cos^n(θ) sinθ cos(nφ)
    # ----------------------------------------------------------------
    f_cor = 2.0 * Omega_frame * sin_lat
    zeta  = (2.0 * omega_rh * sin_lat
             - K * (n+1) * (n+2) * cos_n * sin_lat * np.cos(n * phi))
    eta   = f_cor + zeta

    # ----------------------------------------------------------------
    # Step 3: Balance RHS
    #
    # ∇B = -(f+ζ)(r̂ × v)
    #
    # r̂ × e_phi = e_lat,   r̂ × e_lat = -e_phi
    # => r̂ × v = u_phi*e_lat - u_lat*e_phi
    # => F = -(f+ζ)(r̂×v) = η*(u_lat*e_phi - u_phi*e_lat)
    # ----------------------------------------------------------------
    F_vec = eta[:, None] * (u_lat[:, None] * e_phi - u_phi[:, None] * e_lat)

    # ----------------------------------------------------------------
    # Step 4: Solve ∇²B = ∇·F on the icosahedral mesh
    #
    # Discretization:
    #   div_F[i] = (1/A) Σ_j F_mid · n̂_ij * |edge_ij|
    #   Laplacian[i,j] = w_ij = 1/d_ij²
    # ----------------------------------------------------------------
    area_cell = 4.0 * np.pi * R_sphere**2 / N

    # --- Compute divergence of F ---
    div_F = np.zeros(N)
    for i in range(N):
        ni = n_hat[i]
        for j in neighbors[i]:
            dr = face_centers[j] - face_centers[i]
            d  = np.linalg.norm(dr)
            if d < 1e-12:
                continue
            # Tangential edge direction (outward from i toward j)
            n_edge = dr / d
            n_edge -= (n_edge @ ni) * ni
            n_edge_norm = np.linalg.norm(n_edge)
            if n_edge_norm < 1e-12:
                continue
            n_edge /= n_edge_norm
            # Midpoint flux
            F_mid   = 0.5 * (F_vec[i] + F_vec[j])
            div_F[i] += (F_mid @ n_edge) * d
        div_F[i] /= area_cell

    # --- Build sparse Laplacian ---
    A_mat = lil_matrix((N, N))
    for i in range(N):
        for j in neighbors[i]:
            d_ij = (np.arccos(np.clip(n_hat[i] @ n_hat[j], -1, 1))
                    * R_sphere)
            if d_ij < 1e-12:
                continue
            w = 1.0 / d_ij**2
            A_mat[i, j] +=  w
            A_mat[i, i] -= w

    # Pin cell 0 to set the constant of integration
    rhs       = div_F.copy()
    A_mat[0, :] = 0
    A_mat[0, 0] = 1
    rhs[0]      = 0

    A_csr = A_mat.tocsr()
    B_field, info = minres(A_csr, rhs, rtol=1e-10, maxiter=20000)
    if info != 0:
        print(f'Warning: Poisson solver did not converge (info={info})')
    else:
        print(f'Poisson solver converged')

    # ----------------------------------------------------------------
    # Step 5: Set the integration constant of B
    #
    # Background state: Σ=ρ_0, Π=p_0
    #   h_background = γ/(γ-1) * p_0/ρ_0
    #   B_mean = h_background + <v²>/2
    # ----------------------------------------------------------------
    h_background = gam / (gam - 1) * p_0 / rho_0
    B_mean_target = h_background + np.mean(v_sq) / 2.0
    B_field      += B_mean_target - np.mean(B_field)

    # ----------------------------------------------------------------
    # Step 6: Thermodynamic variables from B
    #
    # h_ent = B - v²/2  (specific enthalpy = γ/(γ-1) Π/Σ)
    # h_ent = γ/(γ-1) K_eos Σ^(γ-1)
    # => Σ = [(γ-1)/(γ K_eos) h_ent]^(1/(γ-1))
    # => Π = K_eos Σ^γ
    # ----------------------------------------------------------------
    h_ent = B_field - v_sq / 2.0

    h_min = 0.1 * h_background
    n_clipped = np.sum(h_ent < h_min)
    if n_clipped > 0:
        print(f'Warning: clipping enthalpy at {n_clipped} cells '
              f'(wave amplitude may be too large)')
    h_ent = np.maximum(h_ent, h_min)

    rho = ((h_ent * (gam - 1)) / (gam * K_eos)) ** (1.0 / (gam - 1))
    p   = K_eos * rho**gam

    # ----------------------------------------------------------------
    # Step 7: Angular momentum and energy
    # ----------------------------------------------------------------
    l = rho[:, None] * np.cross(face_centers, v)   # R_sphere=1
    E = p / (gam - 1) + rho * v_sq / 2.0

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------
    c_s      = np.sqrt(gam * p / rho)
    mach     = np.sqrt(v_sq) / c_s
    rho_var  = (np.max(rho) - np.min(rho)) / rho_0

    # Balance residual: compare ∇B (discrete) with F analytically
    # Use LSQ gradient of B_field
    grad_B = np.zeros((N, 3))
    for i in range(N):
        ni  = n_hat[i]
        ref = np.array([1.,0.,0.]) if abs(ni@[1,0,0]) < 0.9 \
              else np.array([0.,1.,0.])
        e1  = np.cross(ni, ref);  e1 /= np.linalg.norm(e1)
        e2  = np.cross(ni, e1)
        A_lsq, b_lsq, w_lsq = [], [], []
        for j in neighbors[i]:
            dr   = face_centers[j] - face_centers[i]
            dr_t = dr - (dr @ ni) * ni
            w    = 1.0 / (np.linalg.norm(dr_t) + 1e-12)
            A_lsq.append([dr_t @ e1, dr_t @ e2])
            b_lsq.append(B_field[j] - B_field[i])
            w_lsq.append(w)
        A_lsq = np.array(A_lsq);  b_lsq = np.array(b_lsq)
        W_lsq = np.diag(w_lsq)
        AtW   = A_lsq.T @ W_lsq
        coef, _, _, _ = np.linalg.lstsq(AtW @ A_lsq, AtW @ b_lsq, rcond=None)
        grad_B[i] = coef[0]*e1 + coef[1]*e2

    # Residual = |∇B - F|
    balance_err = np.linalg.norm(grad_B - F_vec, axis=1)

    # RH phase speed (Swarztrauber 1996):
    # The wave pattern in the inertial frame rotates at:
    # σ/n = omega_rh - 2(Ω+omega_rh) / (n(n+1)/2 * something)
    # Approximate for omega_rh << Omega:
    # Δω_retardation ≈ 2*Omega / (n*(n+1)/2) = 4*Omega / (n*(n+1))
    # omega_phase_in_frame ≈ omega_rh - 4*Omega/(n*(n+1))
    omega_phase_approx = omega_rh - 4*Omega_frame / (n*(n+1))
    T_wave = abs(2*np.pi / (n * omega_phase_approx)) if omega_phase_approx != 0 else np.inf

    print(f'\n--- Initial data diagnostics ---')
    print(f'Max Mach number          : {np.max(mach):.4f}  (target ~0.1)')
    print(f'Max |v|                  : {np.max(np.sqrt(v_sq)):.6f}')
    print(f'rho variation            : {rho_var*100:.3f}%')
    print(f'p variation              : {(np.max(p)-np.min(p))/p_0*100:.3f}%')
    print(f'EOS residual max         : {np.max(np.abs(p - K_eos*rho**gam)):.2e}')
    print(f'Balance residual (mean)  : {np.mean(balance_err):.3e}')
    print(f'Balance residual (max)   : {np.max(balance_err):.3e}')
    print(f'Inertial period          : {2*np.pi/(2*Omega_frame):.4f} code units')
    print(f'Wave period (approx)     : {T_wave:.4f} code units')
    print(f'  => wave phase speed    : {omega_phase_approx:.5f} rad/unit')

    # ----------------------------------------------------------------
    # Write output
    # ----------------------------------------------------------------
    pd.DataFrame(
        np.column_stack([rho, l[:, 0], l[:, 1], l[:, 2], E])
    ).to_csv('input/input.dat', index=False, sep=' ', header=False)

    return rho, v, p, l, E, B_field


def track_rh_wave_phase(face_centers, p, n_wave=4, lat_band=0.3):
    """
    Diagnostic: locate the longitude of the pressure maximum in a
    latitude band and compare to expected RH phase progression.

    Call this at each output time to measure the phase speed.
    Returns the dominant phase angle of the n-wave Fourier mode.
    """
    n_hat   = face_centers / np.linalg.norm(face_centers, axis=1, keepdims=True)
    sin_lat = n_hat[:, 2]
    phi     = np.arctan2(n_hat[:, 1], n_hat[:, 0])

    # Select cells in target latitude band (near 45°N)
    mask = np.abs(sin_lat - np.sin(np.pi/4)) < lat_band
    if np.sum(mask) < 10:
        return np.nan

    # Fourier projection onto mode n
    p_band  = p[mask] - np.mean(p[mask])
    phi_band = phi[mask]

    A_n = np.mean(p_band * np.cos(n_wave * phi_band))
    B_n = np.mean(p_band * np.sin(n_wave * phi_band))

    phase = np.arctan2(B_n, A_n) / n_wave   # longitude of maximum
    amplitude = np.sqrt(A_n**2 + B_n**2)

    return phase, amplitude

def make_input_5_B():
    gam0=5./3
    gam=2-1/gam0
    #face_centers=pd.read_table('results/face_centers_ico_6.dat', header=None, delimiter=r"\s+")
    face_centers=pd.read_table('results/face_centers.dat', header=None, delimiter=r"\s+")
    N=len(face_centers[0])


    face_centers=np.array(face_centers)


    GM=0.217909
    #rho_0=10
    #p_0=0.01
    rho_0=1e-4
    p_0=1e-6
    omega=np.array([0,0,0.1])
    B0=np.array([0,0,0])

    a_0=np.sqrt(gam*p_0/rho_0)

    M_0=np.linalg.norm(omega)/a_0
    print('M_0= ', M_0)

    #M_0=(gam-1)/p_0 *np.linalg.norm(omega)**2

    rho=np.ones(N, dtype='d')*rho_0 
    p=np.ones(N, dtype='d')*p_0 
    theta=np.arccos((face_centers[:,2])/np.linalg.norm(face_centers, axis=1)) 

    #p=0.5+(np.linalg.norm(omega)**2*rho/2*np.sin(theta)**2)
    #rho=rho_0*(1+(gam-1)/2*M_0**2*np.sin(theta, dtype='d')**2)**(1/(gam-1))
    #p=p_0*(1+(gam-1)/2*M_0**2*np.sin(theta, dtype='d')**2)**(gam/(gam-1))


    l=[]
    v=[]
    B=[]

    
    lon=-np.arccos(face_centers[:,2]/np.linalg.norm(face_centers, axis=1))+np.pi/2 #lat
    phi=np.arctan2(face_centers[:,1]/np.linalg.norm(face_centers, axis=1),face_centers[:,0]/np.linalg.norm(face_centers, axis=1)) #long
    a=1
    R0=a/3
    theta_c=0
    phi_c=0
    r=a*np.arccos(np.sin(theta_c)*np.sin(lon)+np.cos(lon)*np.cos(theta_c)*np.cos(phi-phi_c))
    face_centers=face_centers/np.vstack([np.linalg.norm(face_centers,axis=1),np.linalg.norm(face_centers,axis=1),np.linalg.norm(face_centers,axis=1)]).T

    for face_num, R in enumerate(face_centers):

        if r[face_num]<R0:
            B.append(B0+np.array([0.001,0,0]))
        else:
            B.append(B0)
            #rho[face_num]*=1.2

        # if(theta[face_num]<np.pi/2+0.5+0.3 and theta[face_num]>np.pi/2+0.5-0.3):
        #     l.append(rho[face_num]*np.cross(R,np.cross(omega,R))/(np.linalg.norm(R)**2))
        #     v.append(np.cross(omega,R)/np.linalg.norm(R))
        # else:
        #     l.append(np.array([0,0,0]))
        #     v.append(np.array([0,0,0]))
        l.append(np.array([0,0,0]))
        v.append(np.array([0,0,0]))

    theta=np.arccos(face_centers[:,2]/np.linalg.norm(face_centers, axis=1))
    l=np.array(l) 
    v=np.array(v) 
    B=np.array(B)

    #H=(2*gam0-1)/(gam0-1)*p/(rho*(GM-(np.linalg.norm(v+np.cross(omega, face_centers), axis=1)**2)))
    H=np.ones(N)
    H=np.vstack([H,H,H]).T

    print('mean_Mach= ',np.mean(np.linalg.norm(v, axis=1)/np.sqrt(gam*p/rho)))
    print('max c_s= ',np.max(gam *p/rho))
    if(np.max(gam *p/rho)>1/np.sqrt(3)):
        print('Warning: c_s is too high!')


    B=B*np.sqrt(H)
    #E=1/(gam-1)*p+rho*np.linalg.norm(v, axis=1)*np.linalg.norm(v, axis=1)/2+rho*np.linalg.norm(omega)**2*(np.sin(theta)**2)/2
    E=1/(gam-1)*p+rho*np.linalg.norm(v, axis=1)**2/2+np.linalg.norm(B,axis=1)**2/2
    if(E.any()<0):
        print('Energy<0!!')

    pd.DataFrame(data=np.array([rho, l[:,0],l[:,1],l[:,2],E,B[:,0],B[:,1],B[:,2]]).transpose()).to_csv('input/input.dat',index=False, sep=' ', header=False)



def make_input_5_sp_layer():

    #gam0=4./3
    gam0=5./3
    #gam0=1.002
    gam=2-1/gam0

    #V_conv=0.01
    V_conv=1

    face_centers=pd.read_table('results/face_centers.dat', header=None, delimiter=r"\s+")

    N=len(face_centers[0])
    
    face_centers=np.array(face_centers)
   

    rho=np.ones(N) #10^7 g/cm^2
    #c_s=2*10**(-3)/V_conv 
    c_s=2*10**(-3)/V_conv 
    omega_ns=0


    omega=np.array([0,0,0.01])/V_conv #c

    #omega=np.array([0,0,0.01])/V_conv #c
    


    #============equal entropy initial version=======================================
    
    #rho_0=rho[0]/1e3
    rho_0=rho[0]/6e3
    #rho_0=rho[0]/130

    #p_0=c_s**2*rho_0/gam


    #rho_0=1e-8
    p_0=c_s**2*rho_0/gam
    
    a_0=np.sqrt(gam*p_0/rho_0)
    print("c_s=",a_0)
    M_0=np.linalg.norm(omega)/a_0
    print('Mach_eq=',M_0)
   


    #M_0=(gam-1)/p_0 *np.linalg.norm(omega)**2

    theta=np.arccos(face_centers[:,2]/np.linalg.norm(face_centers, axis=1)) 



    rho=rho_0*(1+(gam-1)/2*M_0**2*np.sin(theta)**2)**(1/(gam-1))
    p=p_0*(1+(gam-1)/2*M_0**2*np.sin(theta)**2)**(gam/(gam-1))

    
    print("p[0]= ", p[0])



    print('rho_avg=',np.sum(rho)/len(face_centers))


    face_centers=np.array(face_centers)

    l=[]
    v=[]
   

    for face_num, R in enumerate(face_centers):
        #==========================================================
        #omega=np.array([0,0,0.03])*np.sin(theta[face_num])**2
        #==========================================================
        l.append(rho[face_num]*np.cross(R,np.cross(omega,R))/(np.linalg.norm(R)**2))
        v.append(np.cross(omega,R)/np.linalg.norm(R))

    print('mean_vel= ',np.mean(np.linalg.norm(v, axis=1)))
    print('mean_Mach= ',np.mean(np.linalg.norm(v, axis=1)/np.sqrt(gam*p/rho)))
    print('max c_s= ',np.max(gam *p/rho))


    l=np.array(l)
    v=np.array(v)

    E=1/(gam-1)*p+rho*((np.linalg.norm(v, axis=1)**2)/2-np.ones(N)*omega_ns**2*(np.sin(theta)**2)/2)
    if(E.any()<0):
        print('Energy<0!!')

    print("E[0]= ", E[0])

    pd.DataFrame(data=np.array([rho, l[:,0],l[:,1],l[:,2],E]).transpose()).to_csv('input/input.dat',index=False, sep=' ', header=False, float_format="%.15f")


def make_input_5_sp_layer_exp():

    #gam0=5./3
    gam0=4./3
    #gam0=1.002
    gam=2-1/gam0

    #V_conv=0.01
    V_conv=1

    face_centers=pd.read_table('results/face_centers.dat', header=None, delimiter=r"\s+")

    N=len(face_centers[0])
    
    face_centers=np.array(face_centers)
   
    turn_angle=0
    turn_matrix=np.matrix([[np.cos(turn_angle),0, np.sin(turn_angle)],[0,1,0],[-np.sin(turn_angle),0, np.cos(turn_angle)]])


    rho=np.ones(N) #10^7 g/cm^2
    #c_s=2*10**(-3)/V_conv 
    c_s=1.2*10**(-4)/V_conv 
    omega_ns=0


    #omega=np.matmul(turn_matrix,np.array([0,0,0.01]))/V_conv #c

    omega=np.array([0,0,0.01])/V_conv #c
    


    #============equal entropy initial version=======================================
    rho_0=rho[0]/100000000000

    #rho_0=rho[0]/130
    #p_0=c_s**2*rho_0/gam


    #rho_0=1e-8
    p_0=c_s**2*rho_0/gam
    
    a_0=np.sqrt(gam*p_0/rho_0)
    print("c_s=",a_0)
    M_0=np.linalg.norm(omega)/a_0
    print('Mach_eq=',M_0)
   


    #M_0=(gam-1)/p_0 *np.linalg.norm(omega)**2



    theta=[]
    for fc in face_centers:
        theta.append(np.arccos( np.matmul(turn_matrix, fc/np.linalg.norm(fc))[0,2] ))
   
    theta=np.array(theta)



    rho=rho_0*(1+(gam-1)/2*M_0**2*np.sin(theta)**2)**(1/(gam-1))
    p=p_0*(1+(gam-1)/2*M_0**2*np.sin(theta)**2)**(gam/(gam-1))

    
    print("p[0]= ", p[0])



    print('rho_avg=',np.sum(rho)/len(face_centers))


    face_centers=np.array(face_centers)

    l=[]
    v=[]

    for face_num, R in enumerate(face_centers):
        #==========================================================
        #omega=np.array([0,0,0.03])*np.sin(theta[face_num])**2
        #==========================================================
        l.append(rho[face_num]*np.cross(R,np.cross(omega,R))/(np.linalg.norm(R)**2))
        v.append(np.cross(omega,R)/np.linalg.norm(R))

    print('mean_vel= ',np.mean(np.linalg.norm(v, axis=1)))
    print('mean_Mach= ',np.mean(np.linalg.norm(v, axis=1)/np.sqrt(gam*p/rho)))
    

    l=np.array(l)
    v=np.array(v)

    E=1/(gam-1)*p+rho*((np.linalg.norm(v, axis=1)**2)/2-np.ones(N)*omega_ns**2*(np.sin(theta)**2)/2)
    if(E.any()<0):
        print('Energy<0!!')

    print("E[0]= ", E[0])

    pd.DataFrame(data=np.array([rho, l[:,0],l[:,1],l[:,2],E]).transpose()).to_csv('input/input.dat',index=False, sep=' ', header=False, float_format="%.15f")



def make_input_5_sp_layer_diff_rot():

    pa=1
   #gam0=4./3
    gam0=5./3
    #gam0=1.002
    gam=2-1/gam0

    #V_conv=0.01
    V_conv=1

    face_centers=pd.read_table('results/face_centers.dat', header=None, delimiter=r"\s+")

    N=len(face_centers[0])
    
    face_centers=np.array(face_centers)
   

    rho=np.ones(N) #10^7 g/cm^2
    #c_s=2*10**(-3)/V_conv  
    #c_s=2*10**(-3)/V_conv 
    c_s=6*10**(-2)/V_conv 
    omega_ns=0


    omega=np.array([0,0,0.01])/V_conv #c

    omega_0=omega
    
    rho_0=rho[0]/1000
    p_0=c_s**2*rho_0/gam
    
    a_0=np.sqrt(gam*p_0/rho_0)
    print("c_s=",a_0)
    M_0=np.linalg.norm(omega)/a_0
    print('Mach_eq=',M_0)
    theta=np.arccos(face_centers[:,2]/np.linalg.norm(face_centers, axis=1)) #+tilt_angle

    rho=rho_0*(1+(gam-1)/(2*pa+2)*M_0**2*np.sin(theta)**(2*pa+2))**(1/(gam-1))
    p=p_0*(1+(gam-1)/(2*pa+2)*M_0**2*np.sin(theta)**(2*pa+2))**(gam/(gam-1))

    print("p[0]= ", p[0])
    print('rho_avg=',np.sum(rho)/len(face_centers))
    face_centers=np.array(face_centers)

    l=[]
    v=[]

    for face_num, R in enumerate(face_centers):
        #==========================================================
        omega=omega_0*np.sin(theta[face_num])**(pa)
        #==========================================================
        l.append(rho[face_num]*np.cross(R,np.cross(omega,R))/(np.linalg.norm(R)**2))
        v.append(np.cross(omega,R)/np.linalg.norm(R))

    print('mean_vel= ',np.mean(np.linalg.norm(v, axis=1)))
    print('mean_Mach= ',np.mean(np.linalg.norm(v, axis=1)/np.sqrt(gam*p/rho)))
    

    l=np.array(l)
    v=np.array(v)

    E=1/(gam-1)*p+rho*((np.linalg.norm(v, axis=1)**2)/2-np.ones(N)*omega_ns**2*(np.sin(theta)**2)/2)
    if(E.any()<0):
        print('Energy<0!!')

    print("E[0]= ", E[0])

    pd.DataFrame(data=np.array([rho, l[:,0],l[:,1],l[:,2],E]).transpose()).to_csv('input/input.dat',index=False, sep=' ', header=False, float_format="%.15f")





def make_input_5_cos_bell():  #adds energy
    gam=1.4
    face_centers=pd.read_table('results/face_centers.dat', header=None, delimiter=r"\s+")
    N=len(face_centers[0])

    face_centers=np.array(face_centers)


    l=[]
    
    theta=-np.arccos(face_centers[:,2])+np.pi/2 #lat
    phi=np.arctan2(face_centers[:,1],face_centers[:,0]) #long


    omega=np.array([0,0,0.5])
    rho=np.ones(N)
    



    v=[]
    a=1
    R0=a/3
    r=a*np.arccos(np.sin(0)*np.sin(theta)+np.cos(theta)*np.cos(0)*np.cos(phi-np.pi*3/2))

    for face_num, R in enumerate(face_centers):
        
        
        if r[face_num]<R0:
            rho[face_num]=1+0.5*np.cos(np.pi*r[face_num]/R0)


        l.append(rho[face_num]*np.cross(R,np.cross(omega,R))/(np.linalg.norm(R)**2))
        v.append(np.cross(omega,R/np.linalg.norm(R)))
    

    l=np.array(l)
    v=np.array(v)


    theta=np.arccos(face_centers[:,2])
    p=1+(np.linalg.norm(omega)**2*rho/2*np.sin(theta)**2)
    E=1/(gam-1)*p+rho*np.linalg.norm(v, axis=1)*np.linalg.norm(v, axis=1)/2

    pd.DataFrame(data=np.array([rho, l[:,0],l[:,1],l[:,2],E]).transpose()).to_csv('input/input.dat',index=False, sep=' ', header=False)

def make_input_5_const_entr():  


    gam0=5/3
    gam=2-1/gam0
    face_centers=pd.read_table('results/face_centers.dat', header=None, delimiter=r"\s+")
    N=len(face_centers[0])

    face_centers=np.array(face_centers)

    l=[]
    theta=-np.arccos((face_centers[:,2])/np.linalg.norm(face_centers, axis=1)) 

    rho_0=5
    p_0=1
    omega=np.array([0,0,0.1])
    a_0=np.sqrt(gam*p_0/rho_0)

    M_0=np.linalg.norm(omega)/a_0

    #M_0=(gam-1)/p_0 *np.linalg.norm(omega)**2

    rho=rho_0*(1+(gam-1)/2*M_0**2*np.sin(theta)**2)**(1/(gam-1))
    p=p_0*(1+(gam-1)/2*M_0**2*np.sin(theta)**2)**(gam/(gam-1))



    v=[]


    for face_num, R in enumerate(face_centers):
        l.append(rho[face_num]*np.cross(R,np.cross(omega,R))/(np.linalg.norm(R)**2))
        v.append(np.cross(omega,R/np.linalg.norm(R)))
    

    l=np.array(l)
    v=np.array(v)

    print('mean_Mach= ',np.mean(np.linalg.norm(v, axis=1)/np.sqrt(gam*p/rho)))

    
    E=1/(gam-1)*p+rho*np.linalg.norm(v, axis=1)*np.linalg.norm(v, axis=1)/2

    pd.DataFrame(data=np.array([rho, l[:,0],l[:,1],l[:,2],E]).transpose()).to_csv('input/input.dat',index=False, sep=' ', header=False)



#make_input_5_new_p()
#make_input_5()
#make_input_5_B()
#make_input_rossby_haurwitz(n_wave=4, omega_rh_frac=0.1, K_frac=0.1)
make_input_5_coriolis()
#make_input_geostrophic_pressure_first2()
#make_input_geostrophic_coriolis()
#make_input_balanced_cyclone()
#make_input_5_const_entr()
#make_input_5_sp_layer()
#make_input_5_sp_layer_exp()
#make_input_5_sp_layer_diff_rot()
#make_input_4()
#make_input_6_cos_bell()




