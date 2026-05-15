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
        label_pr='T [K]'
        a=11e5
        m_alpha=6.65e-24
        k_b=1.3807e-16
        g=0.217909*1e18/(3.3e-5*3.3e-5*a*a)
        data_rho.loc[:,1:]=m_alpha/k_b * (data_p.loc[:,1:]*9e27)**2 / ( g*(data_rho.loc[:,1:]*1e7)**3)
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
    elif(value=='vel_abs'):
        label_pr='Speed'
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
            data_rho.loc[i,1:]=(2*gam-1)/(gam-1)*data_p.loc[i,1:]/(g_eff*rho0)*1e6

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
        c_s_e=3e-2
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
    
    print(min_rho, max_rho)
        

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
                ax[0].set_xlabel(rf'$\\frac{{{phi_sym}}}{{\\sqrt{{2}}}}$', fontsize=25)
                # y_plot = sqrt(2) * sin(theta_lat) where theta_lat is latitude (pi/2 - colat)
                ax[0].set_ylabel(rf'$\\sqrt{{2}}\\,\\sin({lat_sym})$', fontsize=25)
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
                #fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=colorm2),cax=ax[2], orientation='horizontal', label=r"v/c")
                fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=colorm2),cax=ax[2], orientation='horizontal', label=r"v/sqrt(gh)")
                j += 1


            fig.savefig('plots/fig'+"{0:0>4}".format(i)+'.png', bbox_inches='tight',dpi=200)
            plt.clf()
            plt.close()
    



projection_plots('rho', path="results/", min=None, max=None,skipstep=10,remove_avg_omega=False, print_residuals=False, 
              log=False, add_streamplot=False, deltaplot=False, reldeltaplot=False, minv=0, maxv=1, normalized=False, projection='Mercator', tilt_angle=0)#0.0015)


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
        label_pr=r'm, $10^7 \rm g $ '
    elif(value=='p'):
        data_rho=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        label_pr='Pressure'
    elif(value=='Y'):
        data_rho=pd.read_table(path+'Y.dat', header=None, delimiter=r"\s+")
        label_pr='Y'
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
        plot_data.append(np.mean(np.array(data_rho.loc[step,1:])))

    plot_data=np.array(plot_data)

    plt.plot(t*3.3e-5,plot_data)
    plt.xlabel("t,s")
    plt.ylabel("total "+label_pr)
    plt.savefig('plots/integ_plt.png', bbox_inches='tight',dpi=300)
    plt.clf()
    plt.close()


#integrated_plot('Y')






def vel_plot( remove_avg_omega:bool=False, projection:str='Gall-Peters'):
    gam=1.25

    #turn_angle=np.pi/2

    turn_angle=0

    path_to_res='plots/article_sim_mk2/'
    #path_to_res='plots/article plots updated/'
    #path_to_res='results/'
    #path_to_res='plots/big_quad_next/'
    #path_to_res='plots/big_sim/'
    #path_to_res='plots/0.4c crashes/time series/'
    

    data_full=pd.read_table(path_to_res+'final_state.dat', header=None, delimiter=r"\s+")
    data_faces=pd.read_table(path_to_res+'faces.dat', header=None, delimiter=r"\s+", names=['col' + str(x) for x in range(6) ])
    face_centers=pd.read_table(path_to_res+'face_centers.dat', header=None, delimiter=r"\s+")
    data=pd.read_table(path_to_res+'vertices.dat', header=None, delimiter=r"\s+")
    vertices=np.array(data.loc[:,:])
    faces=np.array(data_faces.loc[:,:])

    vel=[]
    p=[]


    turn_matrix=np.matrix([[np.cos(turn_angle),0, np.sin(turn_angle)],[0,1,0],[-np.sin(turn_angle),0, np.cos(turn_angle)]])


    for num, el in enumerate(data_full.loc[:,4]):
        l=np.array([data_full.loc[num,1],data_full.loc[num,2],data_full.loc[num,3]])
        R=np.array(face_centers.loc[num])
        vel.append(np.cross(R,l)/(-np.linalg.norm(R)*data_full.loc[num,0]))
        p.append(data_full.loc[num,0])
        #p.append((data_full.loc[num,4]-data_full.loc[num,0]/2 * np.linalg.norm(vel[num])**2)*(gam-1))


    for ver_num, vertice in enumerate(vertices):
        vertices[ver_num]=np.matmul(turn_matrix,vertices[ver_num])

    #for face_num, face in enumerate(faces):
    #    faces[face_num]=np.matmul(turn_matrix,faces[face_num])


    faces_new=[]

    for face_num, face in enumerate(faces): #trick for variable length of each face (needed for hex meshes)
        faces_new.append(face[~np.isnan(face)].astype(int))

    faces=faces_new


    theta=-np.arccos(vertices[:,2])+np.pi/2
    phi=np.arctan2(vertices[:,1],vertices[:,0])

    if projection == 'Gall-Peters':
        x_plot=phi/(np.sqrt(2)) #projection
        y_plot=np.sqrt(2)*np.sin(theta)
    elif projection == 'Mercator':
        x_plot=phi
        #y_plot=np.log(np.tan((-theta+np.pi/2)/2 + np.pi/4))
        y_plot=-theta+np.pi/2

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


    vel=np.array(vel)

    for vel_num, ve in enumerate(vel):
            vel[vel_num]=np.matmul(turn_matrix,vel[vel_num])

    face_centers=np.array(face_centers)

    for face_cent_num, face_cent in enumerate(face_centers):
            face_centers[face_cent_num]=np.matmul(turn_matrix,face_centers[face_cent_num])



    theta_fc=np.arccos(face_centers[:,2]/np.linalg.norm(face_centers, axis=1))
    phi_fc=np.arctan2(face_centers[:,1]/np.linalg.norm(face_centers, axis=1),
                    face_centers[:,0]/np.linalg.norm(face_centers, axis=1))
    
    #==============================================================
    # omega_z=np.cross(face_centers,vel)[:,2]/np.sin(theta_fc)**2

    # mask_here=np.logical_and(theta_fc>0.05, theta_fc<np.pi-0.05)

    # bins=np.linspace(0.1,np.pi-0.1,200)
    # digitized = np.digitize(theta_fc, bins)
    # digitized_masks=[]
    # for i in range(len(bins)):
    #     digitized_masks.append(digitized == i)


    # bin_omegas=[np.median(omega_z[digitized_masks[i]]) for i in range(len(bins))]      
    # bin_omegas=np.array(bin_omegas)


    # plt.plot(bins, bin_omegas/(3.3e-5*2*np.pi))
    # #plt.scatter(theta_fc[mask_here], omega_z[mask_here]/(3.3e-5*2*np.pi))
    # plt.ylabel(r'Freq, Hz')
    # plt.xlabel(r'$\theta$')
    # plt.savefig('plots/omegas.png', bbox_inches='tight',dpi=250)
    # plt.clf()
    #==============================================================



    x_fc=phi_fc/np.sqrt(2)
    y_fc=np.sin(-theta_fc+np.pi/2)*np.sqrt(2)

    for face_num,face in enumerate(faces): #fix x
        sign_arr=np.sign(x_plot_full[face_num])
        if( (not (0 in sign_arr)) and (1 in sign_arr) and (-1 in sign_arr) and (np.min(np.abs(x_plot_full[face_num])) > 1)):
            for i,element in enumerate(x_plot_full[face_num]):
                if(element < 0):
                    if projection == 'Gall-Peters':
                        x_plot_full[face_num][i]+=2*np.pi/np.sqrt(2)
                    elif projection == 'Mercator':
                        x_plot_full[face_num][i]+=2*np.pi
            #x_fc[face_num]+=2*np.pi/np.sqrt(2)

        
    patches=[]
    for face_num,face in enumerate(faces):
        polygon = Polygon(np.vstack([x_plot_full[face_num], y_plot_full[face_num]]).T,closed=True)
        patches.append(polygon)


    #rd=np.sin(theta_fc)*np.cos(phi_fc)*vel[:,0]+np.sin(theta_fc)*np.sin(phi_fc)*vel[:,1]+np.cos(theta_fc)*vel[:,2]
    theta_d=np.cos(theta_fc)*np.cos(phi_fc)*vel[:,0]+np.cos(theta_fc)*np.sin(phi_fc)*vel[:,1]-np.sin(theta_fc)*vel[:,2]
    phi_d=(-np.sin(phi_fc)*vel[:,0]+np.cos(phi_fc)*vel[:,1])/np.sin(theta_fc)


    if(remove_avg_omega):
        phi_d-=np.average(phi_d)


    for num,ph in enumerate(phi_d):
        if(ph>1e5 or np.isinf(ph)):
            phi_d[num]=0



    xd=phi_d/np.sqrt(2)
    yd=theta_d*np.sqrt(2)*np.cos(theta_fc)


    #ax[0].quiver(x_fc[::3], y_fc[::3],xd[::3],yd[::3])
    X_gr, Y_gr=np.meshgrid(np.linspace(-2.2,2.2, 100),np.linspace(-1.4, 1.4, 100))
    #X_gr, Y_gr=np.meshgrid(np.linspace(np.min(x_fc),np.max(x_fc), 100),np.linspace(np.min(y_fc),np.max(y_fc), 100))

    mask=np.logical_or(np.isnan(xd, where=False),np.isnan(xd, where=False))


    xd_gr=griddata(np.stack([x_fc[mask].T, y_fc[mask].T]).T, xd[mask],(X_gr,Y_gr), method='nearest')
    yd_gr=griddata(np.stack([x_fc[mask].T, y_fc[mask].T]).T, yd[mask],(X_gr,Y_gr), method='nearest')    



    colorm = plt.get_cmap('viridis')

   

    min_p=np.min(p)
    max_p=np.max(p)

    norm = mpl.colors.Normalize(vmin=min_p, vmax=max_p)
    mpl.rcParams.update({'font.size': 22})

    #rho=(np.array(p-min_p)/(max_p-min_p))
    collection = PatchCollection(patches)
    colors=colorm(norm(p))

    colorm2 = plt.get_cmap('inferno')
    v=np.sqrt(xd_gr**2+yd_gr**2)

    #print(xd_gr,xd_gr)
    #print(np.min(v),np.max(v))
    v_min=np.min(v)
    v_max=np.max(v)


    norm2 = mpl.colors.Normalize(vmin=v_min, vmax=v_max)




    fig, ax = plt.subplots(figsize=(18, 10), layout='constrained', nrows=3,height_ratios=[16,1,1])

    plt.subplots_adjust(hspace=10)
    
    fig.suptitle('t='+"{:10.4f}".format(1.3958)+" s")
    ax[0].set_xlabel(r'$\varphi / \sqrt{2}$', fontsize=25)
    ax[0].set_ylabel(r'$\sqrt{2}  \cos(\theta )$', fontsize=25)
    #for face_num,face in enumerate(faces):
        #ax[0].fill(x_plot_full[face_num], y_plot_full[face_num],facecolor=colorm(rho[face_num]),edgecolor =colorm(rho[face_num]))

    
    ax[0].add_collection(collection)
    collection.set_color(colors)
    ax[0].set_xlim([-2.5, 3.4])
    ax[0].set_ylim([-1.5, 1.5])
    #ax[0].quiver(x_fc[::3], y_fc[::3],xd[::3],yd[::3])
        #if(face_num % 20 == 0):
    #for face_num,face in enumerate(faces):
    #    ax[0].arrow(x_fc[face_num],y_fc[face_num],1e-1*xd[face_num],1e-1*yd[face_num],width=0.007, color='grey', alpha=0.9)

    #ax[0].streamplot(X_gr,Y_gr,xd_gr,yd_gr,color=np.sqrt(xd_gr*2*+yd_gr**2), arrowsize=3)

    ax[0].streamplot(X_gr,Y_gr,xd_gr,yd_gr,color=v,norm=norm2, cmap=colorm2, arrowsize=2,density = 1.2)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colorm),cax=ax[1], orientation='horizontal', label=r'$\Sigma$, $10^7 \rm g \ \rm cm^{-2}$ ')
    #fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colorm),cax=ax[1], orientation='horizontal', label=r'Speed, c ')
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=colorm2),cax=ax[2], orientation='horizontal', label=r"v/c")
    fig.savefig('plots/vel_plot.png', bbox_inches='tight',dpi=150)
    plt.clf()
    plt.close()



#vel_plot(remove_avg_omega=True)



def plot_vs_theta(value:str,path:str='results/', skipstep:int=1, ylim_min:float=0, ylim_max:float=0):


    data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
    face_centers=pd.read_table(path+'face_centers.dat', header=None, delimiter=r"\s+")
    face_centers=np.array(face_centers)
    theta_fc=-np.arccos(np.array(face_centers[:,2])/np.linalg.norm(face_centers, axis=1))+np.pi/2
    if(value=='rho'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        label_pr=r'$\Sigma$, $10^7 \rm g \ \rm cm^{-2}$ '
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
        data_beta=pd.read_table(path+'beta.dat', header=None, delimiter=r"\s+")
        label_pr='Entropy'
        data_rho.loc[:,1:]=data_p.loc[:,1:]/(data_rho.loc[:,1:]**  ( (10-3*data_beta.loc[:,1:])/(8-3*data_beta.loc[:,1:]) )   )

    elif(value=='vel_abs'):
        print("speed")
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
    else:
        print("wrong type of plot value")
        return

    maxstep=len(data_rho.loc[:,0])

    if(ylim_min==0 and ylim_max==0):
        ylim_min=np.min(data_rho.loc[:,1:])*0.9
        ylim_max=np.max(data_rho.loc[:,1:])*1.1

    for i in tqdm(range(maxstep)):
        if((i % skipstep)==0 ):
            plt.scatter(theta_fc, data_rho.loc[i,1:], s=2)
            plt.xlabel(r'$\theta$')
            plt.ylabel(label_pr)
            plt.title('t='+"{:.4f}".format(data_rho.loc[i,0]*3.3e-5)+' s')
            plt.ylim([ylim_min, ylim_max])
            plt.savefig('plots/vel_vs_theta_'+"{0:0>4}".format(i)+'.png', bbox_inches='tight',dpi=200)
            plt.clf()
            plt.close()



#plot_vs_theta('vel_abs',path='results/', skipstep=10, ylim_min=0, ylim_max=3e-4)

#plot_vs_theta('vel_abs',path='results/', skipstep=10, ylim_min=0, ylim_max=3e-2)


