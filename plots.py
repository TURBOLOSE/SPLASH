from weakref import ref

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import gc
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import plotly.express as px
from tqdm import tqdm
from scipy.interpolate import griddata


def projection_plots(value:str, path:str='results/', min:float=None, max:float=None, skipstep:int=1, print_residuals:bool=False, remove_avg_omega:bool=False, 
                     log:bool=False, add_streamplot:bool=False,minv:float=None, maxv:float=None, deltaplot:bool=False, reldeltaplot:bool=False, normalized:bool=False, projection:str='Gall-Peters', tilt_angle:float=0):
    #value = rho,p,omega
    
    
    gam=1.25
    skipf=0
    #path='results/'
    #path='plots/article plots updated/'
    #path='plots/article_sim_mk2/'
    #path='plots/big_quad_next/'
    #path='plots/new split test/2 layers/'
    #path='plots/spinup/'


    face_centers=pd.read_table(path+'face_centers.dat', header=None, delimiter=r"\s+")
    face_centers=np.array(face_centers)/(np.array([np.linalg.norm(np.array(face_centers), axis=1),
        np.linalg.norm(np.array(face_centers), axis=1),np.linalg.norm(np.array(face_centers), axis=1)]).T)

    y_fc_rotated = face_centers[:,1]/np.linalg.norm(face_centers, axis=1) * np.cos(tilt_angle) - face_centers[:,2]/np.linalg.norm(face_centers, axis=1) * np.sin(tilt_angle)
    z_fc_rotated = face_centers[:,1]/np.linalg.norm(face_centers, axis=1) * np.sin(tilt_angle) + face_centers[:,2]/np.linalg.norm(face_centers, axis=1) * np.cos(tilt_angle)

    theta_fc = np.acos(z_fc_rotated)
    phi_fc=np.arctan2(y_fc_rotated, face_centers[:,0]/np.linalg.norm(face_centers, axis=1))

    # Axis-label helpers: when tilt_angle != 0, we're plotting in the tilted frame
    # (rotation about +x), so use primed angles to indicate rotated spherical coords.
    has_tilt = not np.isclose(tilt_angle, 0.0)
    phi_sym = r"\varphi'" if has_tilt else r"\varphi"
    # We use latitude lambda := pi/2 - colatitude(theta_fc), so lambda increases northward.
    lat_sym = r"\lambda'" if has_tilt else r"\lambda"


    if(value=='rho'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        label_pr=r'$\Sigma$, $10^7 \rm g \ \rm cm^{-2}$ '
    elif(value=='p'):
        data_rho=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        label_pr='Pressure'
    elif(value=='h_sw'):
        data_rho=pd.read_table(path+'h.dat', header=None, delimiter=r"\s+")
        label_pr='Height'
    elif(value=='omega'):
        data_rho=pd.read_table(path+'omega.dat', header=None, delimiter=r"\s+")
        label_pr='Omega_z'
    elif(value=='gradp'):
        data_rho=pd.read_table(path+'gradp.dat', header=None, delimiter=r"\n").T
        print(data_rho)
        label_pr='Gradient of Pressure'
    elif(value=='vort'):
        data_rho=pd.read_table(path+'curl.dat', header=None, delimiter=r"\s+")
        #label_pr='Vorticity'
        label_pr=r'Vorticity, $\Omega$ '
        data_rho.loc[:,1:] = data_rho.loc[:,1:]/0.22#omega
        #label_pr='Bernoulli integral -1 /R'
    elif(value=='pot_vort'):
        data_vort=pd.read_table(path+'curl.dat', header=None, delimiter=r"\s+")
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        label_pr=r'Potential Vorticity (rel units)'
        rho=data_rho.loc[:,1:]
        omega=0.1
        data_rho.loc[:,1:]=(data_vort.loc[:,1:]+2*omega*np.cos(theta_fc))/rho
    elif(value=='beta'):
        data_rho=pd.read_table(path+'beta.dat', header=None, delimiter=r"\s+")
        label_pr=r'$\beta$ '
    elif(value=='mach'):
        data_rho=pd.read_table(path+'mach.dat', header=None, delimiter=r"\s+", skipfooter=skipf)
        label_pr='v/c_s (Mach number)'
    elif(value=='Y'):
        data_rho=pd.read_table(path+'Y.dat', header=None, delimiter=r"\s+", skipfooter=skipf)
        label_pr='Helium fraction'
    elif(value=='c_s'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        label_pr='Speed of sound'
        data_rho.loc[:,1:]=data_p.loc[:,1:]/data_rho.loc[:,1:]
        data_rho.loc[:,1:]=np.sqrt(1.25*data_rho.loc[:,1:])
    elif(value=='T'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        data_rho.loc[:,:]=data_rho.loc[:,:].astype(float)
        data_p.loc[:,:]=data_p.loc[:,:].astype(float)
        label_pr='T [K]'
        m_alpha=6.65e-24
        k_b=1.3807e-16
        data_rho.loc[:,1:]=m_alpha/(3*k_b) * gam * (data_p.loc[:,1:]*9.0e27)/ ( (data_rho.loc[:,1:]*1.0e7))
        data_rho.loc[:,:]=data_rho.loc[:,:].astype(float)
    elif(value=='T_sw'):
        data_rho=pd.read_table(path+'h.dat', header=None, delimiter=r"\s+")
        label_pr=r'$T[k]$ '
        m_alpha=6.65e-24/3
        k_b=1.3807e-16
        g0=0.217909
        data_rho.loc[:,1:]=m_alpha/k_b*g0*data_rho.loc[:,1:]*9e20
    elif(value=='entropy'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        data_beta=pd.read_table(path+'beta.dat', header=None, delimiter=r"\s+")
        label_pr='Entropy'
        data_rho.loc[:,1:]=data_p.loc[:,1:]/(data_rho.loc[:,1:]**  ( (10-3*data_beta.loc[:,1:])/(8-3*data_beta.loc[:,1:]) )   )
    elif(value=='B'):
        label_pr='B'
        data_Bx=pd.read_table(path+'Bx.dat', header=None, delimiter=r"\s+")
        data_By=pd.read_table(path+'By.dat', header=None, delimiter=r"\s+")
        data_Bz=pd.read_table(path+'Bz.dat', header=None, delimiter=r"\s+")
        data_rho=data_Bx
        maxstep=len(data_rho.loc[:,0])
        for i in range(maxstep):
            B=np.array([data_Bx.loc[i,1:],data_By.loc[i,1:],data_Bz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            data_rho.loc[i,1:]=np.sqrt(B[:,0]*B[:,0]+B[:,1]*B[:,1]+B[:,2]*B[:,2])
    elif(value=='vel_abs' or value=='vel_abs_sw'):
        label_pr='Speed'
        if(value=='vel_abs_sw'):
            data_rho=pd.read_table(path+'h.dat', header=None, delimiter=r"\s+")
        else:
            data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")
        maxstep=len(data_rho.loc[:,0])
        n_faces=len(data_rho.loc[0,:])-1
        data_rho=data_rho.astype(float)

        for i in range(0,maxstep,skipstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            x=np.linalg.norm(np.cross(face_centers,L), axis=1)/rho0
            data_rho.loc[i,1:]=x
    elif(value=='v_phi' or value=='v_phi_sw'):
        label_pr='V_phi'
        if(value=='v_phi_sw'):
            data_rho=pd.read_table(path+'h.dat', header=None, delimiter=r"\s+")
        else:
            data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")
        maxstep=len(data_rho.loc[:,0])
        n_faces=len(data_rho.loc[0,:])-1
        data_rho=data_rho.astype(float)

        ref = np.array([0.0, 0.0, 1.0])
        e_phi = np.cross(np.broadcast_to(ref, face_centers.shape), face_centers)
        e_phi_norm = np.linalg.norm(e_phi, axis=1)
        near_pole = e_phi_norm < 1e-12
        if np.any(near_pole):
            ref2 = np.array([1.0, 0.0, 0.0])
            e_phi[near_pole] = np.cross(np.broadcast_to(ref2, face_centers[near_pole].shape), face_centers[near_pole])
            e_phi_norm[near_pole] = np.linalg.norm(e_phi[near_pole], axis=1)

        e_phi = e_phi / e_phi_norm[:, None]

        for i in range(0,maxstep,skipstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            v=np.cross(face_centers,L)/np.array([rho0,rho0,rho0]).T
            v_phi = np.einsum('ij,ij->i', v, e_phi)
            data_rho.loc[i,1:]=v_phi
    elif(value=='h'):
        label_pr='Altitude [cm]'
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")
        maxstep=len(data_rho.loc[:,0])
        n_faces=len(data_rho.loc[0,:])-1

        for i in range(maxstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            v=np.linalg.norm(np.cross(face_centers,L), axis=1)/rho0
            GM=0.217909
            g_eff=-v**2+GM
            data_rho.loc[i,1:]=gam*data_p.loc[i,1:]/(g_eff*rho0)*1e6

    else:
        print("wrong type of plot value")
        return

    maxstep=len(data_rho.loc[:,0])

    if(print_residuals):
        for i in range(1,maxstep):
            data_rho.loc[i,1:]-=data_rho.loc[0,1:]
            data_rho.loc[i,1:]/=data_rho.loc[0,1:]
        data_rho.loc[0,:]-=data_rho.loc[0,:]
        label_pr+=" residuals"

    if(deltaplot and reldeltaplot):
        print('Both deltaplot and reldeltaplot cannot be on at the same time')
    else:
        if(deltaplot):
            for i in range(1,maxstep):
                data_rho.loc[i,1:]-=data_rho.loc[0,1:]
                data_rho.loc[i,1:]/=data_rho.loc[0,1:]
            #data_rho.loc[1:maxstep,1:]-=data_rho.loc[0:maxstep-1,1:]
                #data_rho.loc[i,1:]/=data_rho.loc[0,1:]
            data_rho.loc[0,1:]-=data_rho.loc[0,1:]
            label_pr+=", delta (relative units)"
        if(reldeltaplot):
            for i in range(1,maxstep):
                data_rho.loc[i,1:]/=data_rho.loc[i-1,1:]
            data_rho.loc[0,1:]/=data_rho.loc[0,1:]
            label_pr+=", relative delta"

    if(log):
        data_rho.loc[:,1:]=np.log10(data_rho.loc[:,1:])
        label_pr1=r'$log_{10}$ of '+label_pr
        label_pr=label_pr1

    if(normalized and not deltaplot and not reldeltaplot):
        data_rho.loc[:,1:]=data_rho.loc[:,1:]/data_rho.loc[0,1:]
        label_pr=label_pr+'/'+label_pr+'_0'

    if(add_streamplot):
        data_dens=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")

        maxstep=len(data_dens.loc[:,0])
        vel=[]
        for i in range(0,maxstep,skipstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_dens.loc[i,1:]
            v0=-np.cross(face_centers,L)/np.array([rho0,rho0,rho0]).T
            v1=v0.copy()
            v1[:,1]=v0[:,1]*np.cos(tilt_angle) - v0[:,2]*np.sin(tilt_angle)
            v1[:,2]=v0[:,1]*np.sin(tilt_angle) + v0[:,2]*np.cos(tilt_angle)
            vel.append(v1)
        del data_dens #deallocating useless memory
        del data_Lx
        del data_Ly
        del data_Lz
        gc.collect()

        vel=np.array(vel)


        if projection == 'Gall-Peters':
            x_fc=phi_fc/np.sqrt(2) #projection (Gall-Peters)
            y_fc=np.sin(-theta_fc+np.pi/2)*np.sqrt(2)
            yd = -np.sqrt(2) * np.sin(theta_fc) * (np.cos(theta_fc)*np.cos(phi_fc)*vel[:,:,0] + np.cos(theta_fc)*np.sin(phi_fc)*vel[:,:,1] - np.sin(theta_fc)*vel[:,:,2])
            xd=1/np.sqrt(2)*((-np.sin(phi_fc)*vel[:,:,0]+np.cos(phi_fc)*vel[:,:,1])/np.sin(theta_fc))
            X_gr, Y_gr=np.meshgrid(np.linspace(-2.2,2.2, 500),np.linspace(-1.4, 1.4, 500))
        elif projection == 'Mercator':
            x_fc=phi_fc #projection (Mercator)
            #y_fc=np.log(np.tan((-theta_fc+np.pi/2)/2 + np.pi/4))
            y_fc=-theta_fc+np.pi/2
            xd = (-np.sin(phi_fc)*vel[:,:,0] + np.cos(phi_fc)*vel[:,:,1]) / np.sin(theta_fc)
            yd = -(np.cos(theta_fc)*np.cos(phi_fc)*vel[:,:,0] + np.cos(theta_fc)*np.sin(phi_fc)*vel[:,:,1] - np.sin(theta_fc)*vel[:,:,2]) / np.sin(theta_fc)
            X_gr, Y_gr=np.meshgrid(np.linspace(-np.pi,np.pi, 500),np.linspace(-np.pi/2, np.pi/2, 500))

        xd_gr=[]
        yd_gr=[]
        for i in range(len(xd)):
            mask=np.logical_or(np.isnan(xd[i], where=False),np.isnan(xd[i], where=False))
            xd_gr.append(griddata(np.stack([x_fc[mask].T, y_fc[mask].T]).T, xd[i][mask],(X_gr,Y_gr), method='nearest'))
            yd_gr.append(griddata(np.stack([x_fc[mask].T, y_fc[mask].T]).T, yd[i][mask],(X_gr,Y_gr), method='nearest'))    


        #c_s_e=0.0529
        #c_s_e=3e-2
        c_s_e=1
        xd_gr=np.array(xd_gr)/c_s_e
        yd_gr=np.array(yd_gr)/c_s_e

        colorm2 = plt.get_cmap('inferno')
        v=np.sqrt(xd_gr**2+yd_gr**2)
        mask=np.logical_and(~np.isnan(v),~np.isinf(v))
        
        v_min_val = minv if minv is not None else np.min(v[mask])
        v_max_val = maxv if maxv is not None else np.max(v[mask])
        norm2 = mpl.colors.Normalize(vmin=v_min_val, vmax=v_max_val)

    data_faces=pd.read_table(path+'faces.dat', header=None, delimiter=r"\s+", names=['col' + str(x) for x in range(6) ])

    data=pd.read_table(path+'vertices.dat', header=None, delimiter=r"\s+")
    vertices=np.array(data.loc[:,:])
    faces=np.array(data_faces.loc[:,:])

    #==============================================================================================

    # theta=-np.arccos(np.array(face_centers)[:,2]/np.linalg.norm(np.array(face_centers), axis=1)) 
    # gam=2-3./4
    # omega=np.array([0,0,5])
    # rho_0=1
    # p_0=1
    # a_0=np.sqrt(gam*p_0/rho_0)
    # M_0=np.linalg.norm(omega)/a_0
    # rho_aa=rho_0*(1+(gam-1)/2*M_0**2*np.sin(theta)**2)**(1/(gam-1))



    # for i in range(maxstep):
    #     data_rho.loc[i,1:len(faces)]=(data_rho.loc[i,1:len(faces)]-rho_aa)/rho_aa
    #     #data_rho.loc[i,1:len(faces)]=data_p.loc[i,1:len(faces)]/data_rho.loc[i,1:len(faces)]**gam

    #==============================================================================================

    faces_new=[]

    for face_num, face in enumerate(faces): #trick for variable length of each face (needed for hex meshes)
        faces_new.append(face[~np.isnan(face)].astype(int))
        #if(np.isnan(face[5])):
            #print(-theta_fc[face_num]+np.pi/2)
    faces=faces_new




    # theta=-np.arccos(vertices[:,2])+np.pi/2
    # phi=np.arctan2(vertices[:,1],vertices[:,0])

    y_v_rotated = vertices[:,1]/np.linalg.norm(vertices, axis=1) * np.cos(tilt_angle) - vertices[:,2]/np.linalg.norm(vertices, axis=1) * np.sin(tilt_angle)
    z_v_rotated = vertices[:,1]/np.linalg.norm(vertices, axis=1) * np.sin(tilt_angle) + vertices[:,2]/np.linalg.norm(vertices, axis=1) * np.cos(tilt_angle)

    theta = -np.acos(z_v_rotated)+np.pi/2
    phi=np.arctan2(y_v_rotated, vertices[:,0]/np.linalg.norm(vertices, axis=1))



    if projection == 'Gall-Peters':
        x_plot=phi/(np.sqrt(2)) #projection (Gall-Peters)
        y_plot=np.sqrt(2)*np.sin(theta)
        xlim=np.array([-2.5, 3.4])
        ylim=np.array([-1.5, 1.5])
        x_shift = 2*np.pi/np.sqrt(2)
    elif projection == 'Mercator':
        x_plot=phi #projection (Mercator)
        #y_plot=np.log(np.tan((-theta+np.pi/2)/2 + np.pi/4))
        y_plot=theta
        xlim=np.array([-np.pi, np.pi]) 
        ylim=np.array([-np.pi/2, np.pi/2])
        x_shift = 2*np.pi

    x_plot_full=[]
    y_plot_full=[]


    for face in faces:
        temp_x=[]
        temp_y=[]
        for face_el in face:
            temp_x.append(x_plot[face_el])
            temp_y.append(y_plot[face_el])
        x_plot_full.append(temp_x)
        y_plot_full.append(temp_y)


    for face_num,face in enumerate(faces): #fix x (Gall-Peters)
        sign_arr=np.sign(x_plot_full[face_num])
        if( (not (0 in sign_arr)) and (1 in sign_arr) and (-1 in sign_arr) and (np.min(np.abs(x_plot_full[face_num])) > 1)):
            for j1,element in enumerate(x_plot_full[face_num]):
                if(element < 0):
                    if projection == 'Gall-Peters':
                        x_plot_full[face_num][j1]+=2*np.pi/np.sqrt(2)
                    elif projection == 'Mercator':
                        x_plot_full[face_num][j1]+=2*np.pi

        patches = []

    for face_num,face in enumerate(faces):
        polygon = Polygon(np.vstack([x_plot_full[face_num], y_plot_full[face_num]]).T,closed=True)
        patches.append(polygon)

   
    #=====================================================
    # face_centers=np.array(face_centers)
    # omega=np.array([0,0,2])
    # #for i in range(maxstep):
    # #    data_rho.loc[i,1:len(faces)]-=1+(np.linalg.norm(omega)**2*1./2*np.sin(-np.arccos(face_centers[:,2]))**2)
                    
    # data_rho.loc[:,1:len(faces)]=data_p.loc[:,1:len(faces)]/data_rho.loc[:,1:len(faces)]**(1.4)
    #=====================================================
    colorm = plt.get_cmap('viridis')

    plot_values=data_rho.iloc[:,1:].to_numpy(dtype=float)

    if(min==None and max==None):
        min_rho=np.min(plot_values[np.isnan(plot_values)==False])
        max_rho=np.max(plot_values[np.isnan(plot_values)==False])
    else:
        min_rho=min
        max_rho=max
    
    #print(min_rho, max_rho)
        

    #min_rho=0
    #min_rho=np.quantile(data_rho.loc[:maxstep,1:len(x_plot)],0.05)
    #max_rho=np.quantile(data_rho.loc[:maxstep,1:len(x_plot)],0.95)

    #data_rho.loc[:,1:]*=2


    norm = mpl.colors.Normalize(vmin=min_rho, vmax=max_rho)
    mpl.rcParams.update({'font.size': 25})

   

    om_mean=0
    t=data_rho.loc[:,0]
    if(remove_avg_omega):
        data_om=pd.read_table(path+'omega.dat', header=None, delimiter=r"\s+")
        data_om=data_om.loc[:,1:]
        om_mean=np.mean(data_om,axis=1)


    j=0
    for i in tqdm(range(maxstep)): #dens
        if((i % skipstep)==0 ):


            if(remove_avg_omega):
                phi=np.arctan2(vertices[:,1],vertices[:,0])-om_mean[i]*t[i]
                k=np.floor((np.pi-phi)/(2*np.pi))
                phi=phi+2*k*np.pi

                if projection == 'Gall-Peters':
                    x_plot=phi/(np.sqrt(2)) #projection
                    x_shift = 2*np.pi/np.sqrt(2)
                elif projection == 'Mercator':
                    x_plot=phi
                    x_shift = 2*np.pi
                
                x_plot_full=[]


                for face in faces:
                    temp_x=[]
                    for face_el in face:
                        temp_x.append(x_plot[face_el])
                    x_plot_full.append(temp_x)


                for face_num,face in enumerate(faces): #fix x
                    sign_arr=np.sign(x_plot_full[face_num])
                    if( (not (0 in sign_arr)) and (1 in sign_arr) and (-1 in sign_arr) and (np.min(np.abs(x_plot_full[face_num])) > 1)):
                        for j1,element in enumerate(x_plot_full[face_num]):
                            if(element < 0):
                                x_plot_full[face_num][j1]+=x_shift


                patches = []

                for face_num,face in enumerate(faces):
                    polygon = Polygon(np.vstack([x_plot_full[face_num], y_plot_full[face_num]]).T,closed=True)
                    patches.append(polygon)

            collection = PatchCollection(patches)
            colors=colorm(norm(data_rho.loc[i,1:len(faces)]))


            if(add_streamplot):
                fig, ax = plt.subplots(figsize=(18, 10), layout='constrained', nrows=3,height_ratios=[16,1,1])
            else:
                fig, ax = plt.subplots(figsize=(16, 10), layout='constrained', nrows=2,height_ratios=[15,1])
            #fig.tight_layout()
            plt.subplots_adjust(hspace=10)
            #rho=(np.array(data_rho.loc[i,1:len(faces)])-min_rho)/(max_rho-min_rho)
            #fig.suptitle('t='+"{:.4f}".format(data_rho.loc[i,0]))

            extras=''
            fig.suptitle('t='+"{:.4f}".format(data_rho.loc[i,0]*3.3e-5)+' s'+extras)

            if projection == 'Gall-Peters':
                ax[0].set_xlabel(rf'$\frac{{{phi_sym}}}{{\sqrt{{2}}}}$', fontsize=25)
                # y_plot = sqrt(2) * sin(theta_lat) where theta_lat is latitude (pi/2 - colat)
                ax[0].set_ylabel(rf'$\sqrt{{2}}\sin({lat_sym})$', fontsize=25)
            elif projection == 'Mercator':
                ax[0].set_xlabel(rf'${phi_sym}$', fontsize=25)
                # Current implementation uses y_plot = latitude (not log-tan), so label as such.
                ax[0].set_ylabel(rf'${lat_sym}$', fontsize=25)

            #collection = PatchCollection(patches)
            ax[0].add_collection(collection)
            collection.set_color(colors)


            ax[0].set_xlim(xlim)
            ax[0].set_ylim(ylim)

            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colorm),cax=ax[1], orientation='horizontal', label=label_pr)

            if(add_streamplot):

                mask = np.sqrt(xd_gr[j]**2+yd_gr[j]**2) > 1e-10 #1e-3 #1e-8  #threshold = 1e-3
                xd_gr_masked = np.where(mask, xd_gr[j], np.nan)
                yd_gr_masked = np.where(mask, yd_gr[j], np.nan)

                #plt.streamplot(x, y, xd_gr_masked, yd_gr_masked)
                ax[0].streamplot(X_gr,Y_gr,xd_gr_masked,yd_gr_masked,color=v[j],norm=norm2, cmap=colorm2, arrowsize=1, density = 1)
                fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=colorm2),cax=ax[2], orientation='horizontal', label=r"v/c")
                #fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=colorm2),cax=ax[2], orientation='horizontal', label=r"v/sqrt(gh)")
                j += 1


            fig.savefig('plots/fig'+"{0:0>4}".format(i)+'.png', bbox_inches='tight',dpi=200)
            plt.clf()
            plt.close()
    



# projection_plots('T', path='results/', min=None, max=None,skipstep=1,remove_avg_omega=False, print_residuals=False, 
#               log=False, add_streamplot=False, deltaplot=False, reldeltaplot=False, minv=0, maxv=0.004, normalized=False, projection='Mercator', tilt_angle=0)#0.0015)

# projection_plots('rho', path='results/', min=None, max=None,skipstep=1,remove_avg_omega=False, print_residuals=False, 
#               log=False, add_streamplot=False, deltaplot=False, reldeltaplot=False, minv=0, maxv=0.002, normalized=False, projection='Gall-Peters', tilt_angle=0)#0.0015)

# data_rho_c=pd.read_table("results/isoth_cycl9/"+'curl.dat', header=None, delimiter=r"\s+")
# data_rho_ac=pd.read_table("results/isoth_ac9/"+'curl.dat', header=None, delimiter=r"\s+")

# projection_plots('mach', path='results/', min=None, max=None,skipstep=10,remove_avg_omega=False, print_residuals=False, 
#                  log=False, add_streamplot=False, deltaplot=False, reldeltaplot=False, minv=None, maxv=1, normalized=False, projection='Mercator')#0.0015)


#projection_plots('vel_abs', print_residuals=False, print_log=False, add_streamplot=False)



def integrated_plot(value): 
    #value = rho,p

    path='results/'
    #path='plots/cooling/'


    if(value=='rho'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        label_pr=r'\Sigma, $10^7 \rm g/cm^2 $ '
    elif(value=='h_sw'):
        data_rho=pd.read_table(path+'h.dat', header=None, delimiter=r"\s+")
        label_pr=r'h'
    elif(value=='p'):
        data_rho=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        label_pr='Pressure'
    elif(value=='Y'):
        data_rho=pd.read_table(path+'Y.dat', header=None, delimiter=r"\s+")
        label_pr='Y'
    elif(value=='T_sw'):
        data_rho=pd.read_table(path+'h.dat', header=None, delimiter=r"\s+")
        label_pr=r'$T_8 [K]$ '
        m_alpha=6.65e-24/3
        k_b=1.3807e-16
        g0=0.217909
        data_rho.loc[:,1:]=m_alpha/k_b*g0*data_rho.loc[:,1:]*9e20/1e8
    elif(value=='vel_abs'):
        label_pr='Speed'
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")
        face_centers=pd.read_table(path+'face_centers.dat', header=None, delimiter=r"\s+")
        maxstep=len(data_rho.loc[:,0])
        n_faces=len(data_rho.loc[0,:])-1
        data_rho=data_rho.astype(float)

        face_centers=np.array(face_centers)/(np.array([np.linalg.norm(np.array(face_centers), axis=1),
        np.linalg.norm(np.array(face_centers), axis=1),np.linalg.norm(np.array(face_centers), axis=1)]).T)

        for i in range(maxstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            x=np.linalg.norm(np.cross(face_centers,L), axis=1)/rho0
            data_rho.loc[i,1:]=x
    else:
        print("wrong type of plot value")
        return
    
    dist = lambda r1,r2: 2*np.arcsin(np.linalg.norm(r1-r2)/2)

    maxstep=len(data_rho.loc[:,0])


    data_faces=pd.read_table(path+'faces.dat', header=None, delimiter=r"\s+", names=['col' + str(x) for x in range(6) ])
    face_centers=pd.read_table(path+'face_centers.dat', header=None, delimiter=r"\s+")

    data=pd.read_table(path+'vertices.dat', header=None, delimiter=r"\s+")
    vertices=np.array(data.loc[:,:])
    faces=np.array(data_faces.loc[:,:])

    surface_areas=[]

    faces_new=[]
    for face_num, face in enumerate(faces): #trick for variable length of each face (needed for hex meshes)
        faces_new.append(face[~np.isnan(face)].astype(int))

    faces=faces_new
    

    for i,face in enumerate(faces):
        surface_areas.append(0)
        for j,face_vert in enumerate(face):
            j1=j+1
            if(j==len(face)-1):
                j1=0
            a=dist(face_centers.loc[i,:], vertices[faces[i][j]])
            b=dist(face_centers.loc[i,:], vertices[faces[i][j1]])
            c=dist(vertices[faces[i][j]], vertices[faces[i][j1]])

            A = np.arccos((np.cos(a) - np.cos(b) * np.cos(c)) / (np.sin(b) * np.sin(c)))
            B = np.arccos((np.cos(b) - np.cos(a) * np.cos(c)) / (np.sin(a) * np.sin(c)))
            C = np.arccos((np.cos(c) - np.cos(b) * np.cos(a)) / (np.sin(b) * np.sin(a)))
            surface_areas[i] += A + B + C - np.pi

    surface_areas=np.array(surface_areas)

    plot_data=[]
    t=np.array(data_rho.loc[:,0])

    for step in range(maxstep):
        #plot_data.append(np.sum(np.array(data_rho.loc[step,1:])*surface_areas)*1e12)
        plot_data.append(np.mean(np.array(data_rho.loc[step,1:])))

    plot_data=np.array(plot_data)

    plt.plot(t*3.3e-5,plot_data)
    plt.xlabel("t,s")
    plt.ylabel("Total "+label_pr)
    plt.savefig('plots/integ_plt.png', bbox_inches='tight',dpi=300)
    plt.clf()
    plt.close()


#integrated_plot('Y')




def plot_vs_theta(value:str,path:str='results/', skipstep:int=1, ylim_min:float=0, ylim_max:float=0, is_log:bool=False):


    data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
    times=np.array(data_rho.loc[:,0])
    face_centers=pd.read_table(path+'face_centers.dat', header=None, delimiter=r"\s+")
    face_centers=np.array(face_centers)
    theta_fc=-np.arccos(np.array(face_centers[:,2])/np.linalg.norm(face_centers, axis=1))+np.pi/2
    if(value=='rho'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        label_pr=r'$\Sigma$, $10^7 \rm g \ \rm cm^{-2}$ '
    elif(value=='h_sw'):
        data_rho=pd.read_table(path+'h.dat', header=None, delimiter=r"\s+")
        label_pr=r'$h$ '
    elif(value=='T_sw'):
        data_rho=pd.read_table(path+'h.dat', header=None, delimiter=r"\s+")
        label_pr=r'T$_8$ [$10^8$ K] '
        m_alpha=6.65e-24
        k_b=1.3807e-16
        g0=0.217909
        data_rho.loc[:,1:]=m_alpha/k_b*g0*data_rho.loc[:,1:]*9e20/3 / 1e8
        #print('log T=', np.log10(np.mean(m_alpha/k_b * g0 * h * 9e20)))
    elif(value=='p'):
        data_rho=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        label_pr='Pressure'
    elif(value=='omega'):
        data_rho=pd.read_table(path+'omega.dat', header=None, delimiter=r"\s+")
        label_pr='Omega_z'
    elif(value=='vort'):
        data_rho=pd.read_table(path+'curl.dat', header=None, delimiter=r"\s+")
        #label_pr='Vorticity'
        label_pr=r'Vorticity, $\Omega$ '
        #label_pr='Bernoulli integral -1 /R'
    elif(value=='beta'):
        data_rho=pd.read_table(path+'beta.dat', header=None, delimiter=r"\s+")
        label_pr=r'$\beta$ '
    elif(value=='Y'):
        data_rho=pd.read_table(path+'Y.dat', header=None, delimiter=r"\s+")
        label_pr=r'$Y$ '
    elif(value=='mach'):
        data_rho=pd.read_table(path+'mach.dat', header=None, delimiter=r"\s+")
        label_pr='Mach number'
    elif(value=='c_s'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        label_pr='Speed of sound'
        data_rho.loc[:,1:]=data_p.loc[:,1:]/data_rho.loc[:,1:]
        data_rho.loc[:,1:]=np.sqrt(1.25*data_rho.loc[:,1:])
    elif(value=='entropy'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        #data_beta=pd.read_table(path+'beta.dat', header=None, delimiter=r"\s+")
        label_pr='Entropy'
        data_rho.loc[:,1:]=data_p.loc[:,1:]/(data_rho.loc[:,1:]** 1.25)  #( (10-3*data_beta.loc[:,1:])/(8-3*data_beta.loc[:,1:]) )   )
    elif(value=='pot_vort'):
        data_vort=pd.read_table(path+'curl.dat', header=None, delimiter=r"\s+")
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        label_pr=r'Potential Vorticity (rel units)'
        rho=data_rho.loc[:,1:]
        omega=0.1
        data_rho.loc[:,1:]=(data_vort.loc[:,1:]+2*omega*np.sin(theta_fc))/rho
    elif(value=='pot_vort_sw'):
        data_vort=pd.read_table(path+'curl.dat', header=None, delimiter=r"\s+")
        data_rho=pd.read_table(path+'h.dat', header=None, delimiter=r"\s+")
        label_pr=r'Potential Vorticity (rel units)'
        rho=data_rho.loc[:,1:]
        omega=0.1
        data_rho.loc[:,1:]=(data_vort.loc[:,1:]+2*omega*np.sin(theta_fc))/rho
    elif(value=='T'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        data_rho.loc[:,:]=data_rho.loc[:,:].astype(float)
        data_p.loc[:,:]=data_p.loc[:,:].astype(float)
        label_pr='T [K]'
        m_alpha=6.65e-24/3
        k_b=1.3807e-16
        data_rho.loc[:,1:]=m_alpha/k_b * 1.25 * (data_p.loc[:,1:]*9.0e27)/ ( (data_rho.loc[:,1:]*1.0e7))
        data_rho.loc[:,:]=data_rho.loc[:,:].astype(float)
    elif(value=='rho3'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        data_rho.loc[:,:]=data_rho.loc[:,:].astype(float)
        data_p.loc[:,:]=data_p.loc[:,:].astype(float)
        label_pr=r'\rho, 10^5 g/cm^3'
        a=10e5
        g=0.217909*1e18/(3.3e-5*3.3e-5*a*a); 
        data_rho.loc[:,1:]=g*data_rho.loc[:,1:]**2*1e14/(1.25*data_p.loc[:,1:]*9e27)/1e5
        data_rho.loc[:,:]=data_rho.loc[:,:].astype(float)
    elif(value=='vel_abs'):
        label_pr='Speed'
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")
        face_centers=pd.read_table(path+'face_centers.dat', header=None, delimiter=r"\s+")
        maxstep=len(data_rho.loc[:,0])
        n_faces=len(data_rho.loc[0,:])-1
        data_rho=data_rho.astype(float)

        face_centers=np.array(face_centers)/(np.array([np.linalg.norm(np.array(face_centers), axis=1),
        np.linalg.norm(np.array(face_centers), axis=1),np.linalg.norm(np.array(face_centers), axis=1)]).T)

        for i in range(0,maxstep,skipstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            x=np.linalg.norm(np.cross(face_centers,L), axis=1)/rho0
            data_rho.loc[i,1:]=x

    elif(value=='RT_cr'):
        label_pr=r'g$_{eff} \cdot d \Sigma /d \theta$ [ g cm$^{-1}$ s$^{-2}$ ]'

        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        maxstep=len(data_rho.loc[:,0])
        n_faces=len(data_rho.loc[0,:])-1
        data_rho=data_rho.astype(float)

        ref = np.array([0.0, 0.0, 1.0])
        e_phi = np.cross(np.broadcast_to(ref, face_centers.shape), face_centers)
        e_phi_norm = np.linalg.norm(e_phi, axis=1)
        near_pole = e_phi_norm < 1e-12
        if np.any(near_pole):
            ref2 = np.array([1.0, 0.0, 0.0])
            e_phi[near_pole] = np.cross(np.broadcast_to(ref2, face_centers[near_pole].shape), face_centers[near_pole])
            e_phi_norm[near_pole] = np.linalg.norm(e_phi[near_pole], axis=1)

        e_phi = e_phi / e_phi_norm[:, None]

        omega=0.1
        R=10e5
        g=0.217909*1e18/(3.3e-5*3.3e-5*R*R); 
        th_fc=np.pi/2-theta_fc

        # Bin by theta (mean within each bin)
        bins=np.linspace(0,np.pi,100)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        eps = 1e-12
        theta_clipped = np.clip(th_fc, bins[0] + eps, bins[-1] - eps)
        bin_idx = np.digitize(theta_clipped, bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < len(bin_centers))
        counts = np.bincount(bin_idx[valid], minlength=len(bin_centers)).astype(float)
        nonempty = counts > 0

        res=[]
        for i in range(0,maxstep,skipstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            v=np.cross(face_centers,L)/np.array([rho0,rho0,rho0]).T
            v_phi = np.einsum('ij,ij->i', v, e_phi)
            geff=(v_phi**2/np.tan(th_fc)+2*omega*v_phi*np.sin(th_fc))
            #geff=v_phi**2/np.tan(th_fc)/R*9e20+2*omega*v_phi*np.sin(th_fc)*3e10/(3.3e-5)
            geff[np.abs(th_fc)<1e-4]=0
            rho3=g*rho0**2*1e14/(1.25*data_p.loc[i,1:]*9e27)/1e5
            #data_rho.loc[i,1:]=geff*np.gradient(rho3, th_fc)
            #data_rho.loc[i,1:]=v_phi

            rho0_sum = np.bincount(bin_idx[valid], weights=rho0[valid], minlength=len(bin_centers))
            rho0_binned = np.full_like(bin_centers, np.nan, dtype=float)
            rho0_binned[nonempty] = rho0_sum[nonempty] / counts[nonempty]

            geff_sum = np.bincount(bin_idx[valid], weights=geff[valid], minlength=len(bin_centers))
            geff_binned = np.full_like(bin_centers, np.nan, dtype=float)
            geff_binned[nonempty] = geff_sum[nonempty] / counts[nonempty]


            res.append(-geff_binned*np.gradient(rho0_binned, bin_centers)*1e7*3e10/(3.3e-5))
            theta_fc=-bin_centers+np.pi/2


    elif(value=='N2'):
        label_pr=r'N$^2$ [ s$^{-2}$ ]'

        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        maxstep=len(data_rho.loc[:,0])
        n_faces=len(data_rho.loc[0,:])-1
        data_rho=data_rho.astype(float)

        ref = np.array([0.0, 0.0, 1.0])
        e_phi = np.cross(np.broadcast_to(ref, face_centers.shape), face_centers)
        e_phi_norm = np.linalg.norm(e_phi, axis=1)
        near_pole = e_phi_norm < 1e-12
        if np.any(near_pole):
            ref2 = np.array([1.0, 0.0, 0.0])
            e_phi[near_pole] = np.cross(np.broadcast_to(ref2, face_centers[near_pole].shape), face_centers[near_pole])
            e_phi_norm[near_pole] = np.linalg.norm(e_phi[near_pole], axis=1)

        e_phi = e_phi / e_phi_norm[:, None]

        omega=0.1
        R=10e5
        g=0.217909*1e18/(3.3e-5*3.3e-5*R*R); 
        th_fc=np.pi/2-theta_fc


        # Bin by theta (mean within each bin)
        bins=np.linspace(0,np.pi,100)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        eps = 1e-12
        theta_clipped = np.clip(th_fc, bins[0] + eps, bins[-1] - eps)
        bin_idx = np.digitize(theta_clipped, bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < len(bin_centers))
        counts = np.bincount(bin_idx[valid], minlength=len(bin_centers)).astype(float)

        res=[]
        for i in range(0,maxstep,skipstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            v=np.cross(face_centers,L)/np.array([rho0,rho0,rho0]).T
            v_phi = np.einsum('ij,ij->i', v, e_phi)
            #geff=(v_phi**2/np.tan(th_fc)+2*omega*v_phi*np.sin(th_fc)) * 3e10/(3.3e-5)
            geff=(v_phi**2/np.tan(th_fc)+2*omega*v_phi*np.sin(th_fc))
            geff[np.logical_or(np.abs(th_fc)<1e-4, np.abs(th_fc-np.pi)<1e-4)] = 0
            rho3=g*rho0**2*1e14/(1.4*data_p.loc[i,1:]*9e27)/1e5
            #data_rho.loc[i,1:]=-geff*np.gradient(rho3, th_fc)/(rho3) /(3.3e-5)**2
            entr=data_p.loc[i,1:]/(rho0**1.4)
            nonempty = counts > 0

            entr_sum = np.bincount(bin_idx[valid], weights=entr[valid], minlength=len(bin_centers))
            entr_binned = np.full_like(bin_centers, np.nan, dtype=float)
            entr_binned[nonempty] = entr_sum[nonempty] / counts[nonempty]

            geff_sum = np.bincount(bin_idx[valid], weights=geff[valid], minlength=len(bin_centers))
            geff_binned = np.full_like(bin_centers, np.nan, dtype=float)
            geff_binned[nonempty] = geff_sum[nonempty] / counts[nonempty]
            #data_rho.loc[i,1:]=-geff*np.gradient(entr, th_fc)/(entr*1.4) /(3.3e-5)**2
            #data_rho.loc[i,1:]=-geff*np.gradient(rho0, th_fc)/(rho0) /(3.3e-5)**2

            res.append(-geff_binned*np.gradient(entr_binned, bin_centers)/(entr_binned*1.4) /(3.3e-5)**2)
            theta_fc=-bin_centers+np.pi/2


    elif(value=='Fr'):
       #label_pr=r'Froude number'
        label_pr=r'Ro'
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        maxstep=len(data_rho.loc[:,0])
        n_faces=len(data_rho.loc[0,:])-1
        data_rho=data_rho.astype(float)

        ref = np.array([0.0, 0.0, 1.0])
        e_phi = np.cross(np.broadcast_to(ref, face_centers.shape), face_centers)
        e_phi_norm = np.linalg.norm(e_phi, axis=1)
        near_pole = e_phi_norm < 1e-12
        if np.any(near_pole):
            ref2 = np.array([1.0, 0.0, 0.0])
            e_phi[near_pole] = np.cross(np.broadcast_to(ref2, face_centers[near_pole].shape), face_centers[near_pole])
            e_phi_norm[near_pole] = np.linalg.norm(e_phi[near_pole], axis=1)

        e_phi = e_phi / e_phi_norm[:, None]

        omega=0.1
        R=10e5
        g=0.217909*1e18/(3.3e-5*3.3e-5*R*R); 
        th_fc=np.pi/2-theta_fc

        res=[]
        for i in range(0,maxstep,skipstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            v=np.cross(face_centers,L)/np.array([rho0,rho0,rho0]).T
            v_phi = np.einsum('ij,ij->i', v, e_phi)
            c_s=np.sqrt(1.4*data_p.loc[i,1:]/rho0)
            #res.append(np.abs(v_phi)/c_s)
            #res.append(0.4/(c_s/np.abs(2*omega*np.cos(th_fc)))) #L_j/LD
            res.append(np.abs(v_phi)/np.abs(0.4*2*omega*np.cos(th_fc))) #Ro


    elif(value=='h'):
        label_pr='Altitude [cm]'
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        maxstep=len(data_rho.loc[:,0])
        for i in range(maxstep):
            rho0=data_rho.loc[i,1:]
            GM=0.217909
            g_eff=GM
            data_rho.loc[i,1:]=1.25*data_p.loc[i,1:]/(g_eff*rho0)*1e6
    else:
        print("wrong type of plot value")
        return

    maxstep=len(data_rho.loc[:,0])




    shift_indexing = True
    if(not ('res' in locals() or 'res' in globals())):
        res=np.array(data_rho.loc[:,1:])
        shift_indexing = False


    if(ylim_min==0 and ylim_max==0):
        ylim_min=np.min(res)*0.9
        ylim_max=np.max(res)*1.1


    for i in tqdm(range(0,maxstep, skipstep)):
        j=i
        if(shift_indexing):
            j=i//skipstep

        plt.scatter(theta_fc, res[j], s=4)
        #plt.plot(theta_fc, 1/np.cos(theta_fc))
        plt.xlabel(r'$\theta$')
        plt.ylabel(label_pr)
        if(is_log):
            plt.yscale('log')
        plt.title('t='+"{:.4f}".format(times[i]*3.3e-5)+' s')
        plt.ylim([ylim_min, ylim_max])
        plt.savefig('plots/T_vs_theta_'+"{0:0>4}".format(i)+'.png', bbox_inches='tight',dpi=200)
        plt.clf()
        plt.close()


plot_vs_theta('Fr',path='results/', skipstep=5, ylim_min=-0.1, ylim_max=1)#, is_log=True)


#plot_vs_theta('vel_abs',path='results/polar_burst_RT?/', skipstep=5, ylim_min=0, ylim_max=0.03)

#plot_vs_theta('pot_vort_sw',path="results/", skipstep=1, ylim_min=-15000, ylim_max=15000,is_log=False)
#plot_vs_theta('vel_abs',path='results/', skipstep=10, ylim_min=0, ylim_max=1e-3)
#plot_vs_theta('omega',path='results/burst_1e6_2e8/', skipstep=5, ylim_min=0, ylim_max=0,is_log=True)


