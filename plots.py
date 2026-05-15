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
                     log:bool=False, add_streamplot:bool=False, deltaplot:bool=False, reldeltaplot:bool=False, skiprows:int=0, total_steps:int=0, tilt_angle:float=0): 
    #value = rho,p,omega
    
    if(total_steps==0):
        total_steps=int(1e10) #just a big number for the case when user doesn't want to set it, so that all steps are plotted

    gam=1.25
    skipf=0
    sqrt2=np.sqrt(2)
    twopi_over_sqrt2=2*np.pi/sqrt2
    #path='results/'
    #path='plots/article plots updated/'
    #path='plots/article_sim_mk2/'
    #path='plots/big_quad_next/'
    #path='plots/new split test/2 layers/'
    #path='plots/spinup/'


    face_centers=pd.read_table(path+'face_centers.dat', header=None, delimiter=r"\s+")
    face_centers=np.array(face_centers)/(np.array([np.linalg.norm(np.array(face_centers), axis=1),
        np.linalg.norm(np.array(face_centers), axis=1),np.linalg.norm(np.array(face_centers), axis=1)]).T)

    theta_fc = np.acos(face_centers[:,1]/np.linalg.norm(face_centers, axis=1) * np.sin(tilt_angle) + face_centers[:,2]/np.linalg.norm(face_centers, axis=1) * np.cos(tilt_angle))
    #theta_fc=np.arccos(face_centers[:,2]/np.linalg.norm(face_centers, axis=1))
    phi_fc=np.arctan2(face_centers[:,1],face_centers[:,0])




    if(value=='rho'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr=r'$\Sigma$, $10^7 \rm g \ \rm cm^{-2}$ '
    elif(value=='p'):
        data_rho=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr='Pressure'
    elif(value=='omega'):
        data_rho=pd.read_table(path+'omega.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr='Omega_z'
    elif(value=='vort'):
        data_rho=pd.read_table(path+'curl.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        #label_pr='Vorticity'
        label_pr=r'Vorticity, $\Omega$ '
        #label_pr='Bernoulli integral -1 /R'
    elif(value=='beta'):
        data_rho=pd.read_table(path+'beta.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr=r'$\beta$ '
    elif(value=='mach'):
        data_rho=pd.read_table(path+'mach.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr='Mach number'
    elif(value=='c_s'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr='Speed of sound'
        data_rho.loc[:,1:]=data_p.loc[:,1:]/data_rho.loc[:,1:]
        data_rho.loc[:,1:]=np.sqrt(gam*data_rho.loc[:,1:])
    elif(value=='entropy'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_beta=pd.read_table(path+'beta.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr='Entropy'
        data_rho.loc[:,1:]=data_p.loc[:,1:]/(data_rho.loc[:,1:]**  ( (10-3*data_beta.loc[:,1:])/(8-3*data_beta.loc[:,1:]) )   )
    elif(value=='vel_abs'):
        label_pr='Speed'
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+", skiprows=skiprows)

        maxstep=np.min([len(data_rho.loc[:,0]), total_steps])


        for i in range(0,maxstep,skipstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            data_rho.loc[i,1:]=np.linalg.norm(np.cross(face_centers,L), axis=1)/rho0
    elif(value=='h'):
        label_pr='H [cm]'
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        GM = 0.217909 # grav parameter in R_unit^3/t_unit^2
        maxstep=np.min([len(data_rho.loc[:,0]), total_steps])


        e_theta=np.column_stack([np.cos(theta_fc)*np.cos(phi_fc), np.cos(theta_fc)*np.sin(phi_fc), -np.sin(theta_fc)])

        # Keep current behavior: the final assignment overwrites all rows using the last step values.
        for i in tqdm(range(0,maxstep,skipstep)):
            L=np.column_stack([
                data_Lx.iloc[i,1:].to_numpy(dtype=float),
                data_Ly.iloc[i,1:].to_numpy(dtype=float),
                data_Lz.iloc[i,1:].to_numpy(dtype=float)
            ])
            rho0=data_rho.iloc[i,1:].to_numpy(dtype=float)
            vel_i=np.cross(face_centers,L)/rho0[:,None]
            g_eff =   GM - np.linalg.norm(vel_i, axis=1)**2 
            data_rho.loc[i,1:]=(2*gam-1)*(gam-1)*data_p.iloc[i,1:]/(data_rho.iloc[i,1:]*g_eff) * 1e6

    elif value=='L':
        label_pr=r'T (K)'
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")

        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")

        data_beta=pd.read_table(path+'beta.dat', header=None, delimiter=r"\s+")

        #maxstep=np.min([len(data_rho.loc[:,0]), len(data_p.loc[:,0]),len(data_Lx.loc[:,0]),len(data_Ly.loc[:,0]),len(data_Lz.loc[:,0]), len(data_beta.loc[:,0])])
        maxstep=np.min([len(data_rho.loc[:,0]), total_steps])


        data_beta=np.array(data_beta.loc[:,1:])
        vel=[]

        for i in tqdm(range(0,maxstep,skipstep)):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            vel.append(np.linalg.norm(np.cross(face_centers,L), axis=1)/rho0)
        vel=np.array(vel)


        surface_area=4*np.pi/len(face_centers)
        kappa = 3.4e6 #// scattering opacity in 1/Sigma_unit (R_unit^2/M_unit)
        GM = 0.217909 # grav parameter in R_unit^3/t_unit^2
        g_eff =  - vel * vel + GM
        j=0
        for step in tqdm(range(0,maxstep,skipstep)):
            #data_rho.loc[step,1:]=(g_eff[j] / kappa * (1 - data_beta[step])* surface_area)*9e27*1e12/(3.3e-5) /4e33 
            data_rho.loc[step,1:]=np.power((g_eff[j] / kappa * (1 - data_beta[step])* surface_area)*9e27*1e12/(3.3e-5) /(5.6e-5),1/4) 
            j+=1


    maxstep=np.min([len(data_rho.loc[:,0]), total_steps])

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
                data_rho.loc[i,1:]-=data_rho.loc[i-1,1:]
            #data_rho.loc[1:maxstep,1:]-=data_rho.loc[0:maxstep-1,1:]
                #data_rho.loc[i,1:]/=data_rho.loc[0,1:]
            data_rho.loc[0,1:]-=data_rho.loc[0,1:]
            label_pr+=", delta"
        if(reldeltaplot):
            for i in range(1,maxstep):
                data_rho.loc[i,1:]/=data_rho.loc[i-1,1:]
            data_rho.loc[0,1:]/=data_rho.loc[0,1:]
            label_pr+=", relative delta"

    if(log):
        data_rho.loc[:,1:]=np.log10(data_rho.loc[:,1:])
        label_pr1=r'$log_{10}$ of '+label_pr
        label_pr=label_pr1

    


    if(add_streamplot):
        data_dens=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+", skiprows=skiprows)

        maxstep=len(data_dens.loc[:,0])
        vel=[]
        for i in range(0,maxstep,skipstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_dens.loc[i,1:]
            vel.append(-np.cross(face_centers,L)/np.array([rho0,rho0,rho0]).T)
        del data_dens #deallocating useless memory
        del data_Lx
        del data_Ly
        del data_Lz
        gc.collect()

        vel=np.array(vel)
        #theta_fc=np.arccos(face_centers[:,2])
        x_fc=phi_fc/np.sqrt(2) #Projection
        y_fc=np.sin(-theta_fc+np.pi/2)*np.sqrt(2)

        yd=np.sqrt(2)*np.sin(theta_fc)*(np.cos(theta_fc)*np.cos(phi_fc)*vel[:,:,0]+np.cos(theta_fc)*np.sin(phi_fc)*vel[:,:,1]-np.sin(theta_fc)*vel[:,:,2])
        xd=1/np.sqrt(2)*((-np.sin(phi_fc)*vel[:,:,0]+np.cos(phi_fc)*vel[:,:,1])/np.sin(theta_fc))

        X_gr, Y_gr=np.meshgrid(np.linspace(-2.2,2.2, 200),np.linspace(-1.4, 1.4, 200))

        xd_gr=[]
        yd_gr=[]
        for i in range(0,len(xd)):
            #mask=np.logical_or(np.isnan(xd[i], where=False),np.isnan(xd[i], where=False))
            mask = ~np.logical_or(np.logical_or(np.isnan(xd[i]),np.isinf(xd[i])), np.logical_or(np.isnan(yd[i]),np.isinf(yd[i])))
            xd_gr.append(griddata(np.stack([x_fc[mask].T, y_fc[mask].T]).T, xd[i][mask],(X_gr,Y_gr), method='nearest'))
            yd_gr.append(griddata(np.stack([x_fc[mask].T, y_fc[mask].T]).T, yd[i][mask],(X_gr,Y_gr), method='nearest'))    

        xd_gr=np.array(xd_gr)
        yd_gr=np.array(yd_gr)

        colorm2 = plt.get_cmap('inferno')
        v=np.sqrt(xd_gr**2+yd_gr**2)
        mask=~np.isnan(v)
        norm2 = mpl.colors.Normalize(vmin=np.abs(np.min(v[mask])), vmax=np.abs(np.max(v[mask])))




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




    #theta=-np.arccos(vertices[:,2])+np.pi/2
    theta = -np.acos(vertices[:,1]/np.linalg.norm(vertices, axis=1) * np.sin(tilt_angle) + vertices[:,2]/np.linalg.norm(vertices, axis=1) * np.cos(tilt_angle))+np.pi/2
    phi=np.arctan2(vertices[:,1],vertices[:,0])


    x_plot=phi/sqrt2 #projection
    y_plot=sqrt2*np.sin(theta)


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


    for face_num,face in enumerate(faces): #fix x
        sign_arr=np.sign(x_plot_full[face_num])
        if( (not (0 in sign_arr)) and (1 in sign_arr) and (-1 in sign_arr) and (np.min(np.abs(x_plot_full[face_num])) > 1)):
            for i,element in enumerate(x_plot_full[face_num]):
                if(element < 0):
                    x_plot_full[face_num][i]+=twopi_over_sqrt2


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
        print(min_rho, max_rho)
    else:
        min_rho=min
        max_rho=max
        
        

    #min_rho=0
    #min_rho=np.quantile(data_rho.loc[:maxstep,1:len(x_plot)],0.05)
    #max_rho=np.quantile(data_rho.loc[:maxstep,1:len(x_plot)],0.95)

    #data_rho.loc[:,1:]*=2


    norm = mpl.colors.Normalize(vmin=min_rho, vmax=max_rho)
    mpl.rcParams.update({'font.size': 25})

   

    om_mean=0
    t=data_rho.iloc[:,0].to_numpy(dtype=float)
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


                x_plot=phi/sqrt2 #projection


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
                                x_plot_full[face_num][j1]+=twopi_over_sqrt2


                patches = []

                for face_num,face in enumerate(faces):
                    polygon = Polygon(np.vstack([x_plot_full[face_num], y_plot_full[face_num]]).T,closed=True)
                    patches.append(polygon)

            collection = PatchCollection(patches)
            colors=colorm(norm(plot_values[i]))


            if(add_streamplot):
                fig, ax = plt.subplots(figsize=(18, 10), layout='constrained', nrows=3,height_ratios=[16,1,1])
            else:
                fig, ax = plt.subplots(figsize=(16, 10), layout='constrained', nrows=2,height_ratios=[15,1])
            #fig.tight_layout()
            plt.subplots_adjust(hspace=10)
            #rho=(np.array(data_rho.loc[i,1:len(faces)])-min_rho)/(max_rho-min_rho)
            #fig.suptitle('t='+"{:.4f}".format(data_rho.loc[i,0]))
            fig.suptitle('t='+"{:.4f}".format(t[i]*3.3e-5)+' s')
            ax[0].set_xlabel(r'$\varphi / \sqrt{2}$', fontsize=25)
            ax[0].set_ylabel(r'$\sqrt{2}  \cos(\theta )$', fontsize=25)

            #collection = PatchCollection(patches)
            ax[0].add_collection(collection)
            collection.set_color(colors)
            ax[0].set_xlim([-2.5, 3.4])
            ax[0].set_ylim([-1.5, 1.5])



            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colorm),cax=ax[1], orientation='horizontal', label=label_pr)

            if(add_streamplot):
                ax[0].streamplot(X_gr,Y_gr,xd_gr[j],yd_gr[j],color=v[j],norm=norm2, cmap=colorm2, arrowsize=3, density=1.5)
                fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=colorm2),cax=ax[2], orientation='horizontal', label=r"v/c")
                j+=1


            fig.savefig('plots/fig'+"{0:0>4}".format(i)+'.png', bbox_inches='tight',dpi=200)
            plt.close(fig)
    



projection_plots("rho", path='plots/res26_3/', min=None, max=None,skipstep=1000,remove_avg_omega=False, print_residuals=False, 
                 log=False, add_streamplot=True, deltaplot=False, reldeltaplot=False, skiprows=0,total_steps=0, tilt_angle=0.1)



#projection_plots('vel_abs', print_residuals=False, print_log=False, add_streamplot=False)



def integrated_plot(value:str, path:str='results/', kinetic_component:str='total', relative:bool=False): 
    #value = rho,p

    gam=2-1/(5/3)
    #path='results/'
    #path='plots/cooling/'



    if(value=='rho'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        label_pr=r'm, $10^7$ g '
    elif(value=='p'):
        data_rho=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        label_pr='Internal energy [erg]'
        data_rho.loc[:,1:]*=1/(gam-1)
        data_rho.loc[:,1:]*=9*10**27*10**12
    elif(value=='omega'):
        data_rho=pd.read_table(path+'omega.dat', header=None, delimiter=r"\s+")
        label_pr='Omega_z'
    elif(value=='E_kin'):
        valid_components={'total', 'phi', 'theta', 'all'}
        if kinetic_component not in valid_components:
            print("wrong kinetic_component, use 'total', 'phi', 'theta' or 'all'")
            return

        label_pr='Kinetic energy [erg]'
        vel=[]
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")
        face_centers=pd.read_table(path+'face_centers.dat', header=None, delimiter=r"\s+")
        maxstep=len(data_rho.loc[:,0])

        face_centers=np.array(face_centers)
        face_centers_norm=np.linalg.norm(face_centers, axis=1)
        face_centers=face_centers/np.array([face_centers_norm, face_centers_norm, face_centers_norm]).T

        theta_fc=np.arccos(face_centers[:,2]/np.linalg.norm(face_centers, axis=1))
        phi_fc=np.arctan2(face_centers[:,1]/np.linalg.norm(face_centers, axis=1),
                        face_centers[:,0]/np.linalg.norm(face_centers, axis=1))
        e_theta=np.column_stack([np.cos(theta_fc)*np.cos(phi_fc), np.cos(theta_fc)*np.sin(phi_fc), -np.sin(theta_fc)])
        e_phi=np.column_stack([-np.sin(phi_fc), np.cos(phi_fc), np.zeros_like(phi_fc)])

        kinetic_fields={
            'total': np.zeros((maxstep, len(face_centers))),
            'theta': np.zeros((maxstep, len(face_centers))),
            'phi': np.zeros((maxstep, len(face_centers)))
        }

        for i in range(maxstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            vel_i=np.cross(face_centers,L)/np.array([rho0, rho0, rho0]).T
            vel.append(vel_i)

            v_theta=np.sum(vel_i*e_theta, axis=1)
            v_phi=np.sum(vel_i*e_phi, axis=1)

            # v_r is assumed to be zero for this model, so only tangential parts are used.
            kinetic_fields['theta'][i]=data_rho.loc[i,1:]*v_theta**2/2
            kinetic_fields['phi'][i]=data_rho.loc[i,1:]*v_phi**2/2
            kinetic_fields['total'][i]=kinetic_fields['theta'][i]+kinetic_fields['phi'][i]

        conv=9*10**27*10**12
        for key in kinetic_fields:
            kinetic_fields[key]*=conv

        if(kinetic_component!='all'):
            data_rho.loc[:,1:]=kinetic_fields[kinetic_component]
            label_pr=f"Kinetic energy ({kinetic_component}) [erg]"


    elif(value=='E'):
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        label_pr='Energy [erg]'
        v=[]
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")
        face_centers=pd.read_table(path+'face_centers.dat', header=None, delimiter=r"\s+")
        maxstep=len(data_rho.loc[:,0])
        n_faces=len(data_rho.loc[0,:])-1

        face_centers=np.array(face_centers)/(np.array([np.linalg.norm(np.array(face_centers), axis=1),
        np.linalg.norm(np.array(face_centers), axis=1),np.linalg.norm(np.array(face_centers), axis=1)]).T)

        for i in range(maxstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            v.append(np.linalg.norm(np.cross(face_centers,L), axis=1)/rho0)

        v=np.array(v)
        data_rho.loc[:,1:]*=v**2/2
        data_rho.loc[:,1:]+=data_p.loc[:,1:]/(gam-1)
        data_rho.loc[:,1:]*=9*10**27*10**12


    elif(value=='vel_abs'):
        label_pr='Speed'
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")
        face_centers=pd.read_table(path+'face_centers.dat', header=None, delimiter=r"\s+")
        maxstep=len(data_rho.loc[:,0])
        n_faces=len(data_rho.loc[0,:])-1

        face_centers=np.array(face_centers)/(np.array([np.linalg.norm(np.array(face_centers), axis=1),
        np.linalg.norm(np.array(face_centers), axis=1),np.linalg.norm(np.array(face_centers), axis=1)]).T)

        for i in range(maxstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            data_rho.loc[i,1:]=np.linalg.norm(np.cross(face_centers,L), axis=1)/rho0
    else:
        print("wrong type of plot value")
        return
    

    maxstep=len(data_rho.loc[:,0])



    
    dist = lambda r1,r2: 2*np.arcsin(np.linalg.norm(r1-r2)/2)



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

    t=np.array(data_rho.loc[:,0])

    if(value=='E_kin' and kinetic_component=='all'):
        series={}
        for key in ['total','theta','phi']:
            series[key]=np.sum(kinetic_fields[key]*surface_areas[None,:], axis=1)
            np.save(path+f"Kinetic energy ({key}) [erg]", np.column_stack((t*3.3e-5,series[key])))

        plt.plot(t*3.3e-5,series['total'], label='total')
        plt.plot(t*3.3e-5,series['theta'], label='theta')
        plt.plot(t*3.3e-5,series['phi'], label='phi')
        plt.xlabel("t,s")
        plt.ylabel("Total Kinetic energy [erg]")
        plt.legend()
        plt.savefig('plots/1integ_e_kin_split.png', bbox_inches='tight',dpi=300)
        plt.clf()
        plt.close()
    elif(value=='omega'):
        plot_data=[]
        for step in tqdm(range(maxstep)):
            if(relative and value!='E_kin'):
                plot_data.append(np.mean(np.array(data_rho.loc[step,1:]/data_rho.loc[0,1:])))
            else:
                plot_data.append(np.mean(np.array(data_rho.loc[step,1:])))
        plt.plot(t*3.3e-5,plot_data)
        plt.xlabel("t,s")
        plt.ylabel("Total "+label_pr)
        plt.savefig('plots/integ_plot_'+label_pr+'.png', bbox_inches='tight',dpi=300)
        print('plots/integ_plot_'+label_pr+'.png')
        plt.clf()
        plt.close()
    else:
        plot_data=[]
        for step in tqdm(range(maxstep)):
            if(relative and value!='E_kin'):
                plot_data.append(np.sum(np.array(data_rho.loc[step,1:])*surface_areas/(4*np.pi)))
            else:
                plot_data.append(np.sum(np.array(data_rho.loc[step,1:])*surface_areas/(4*np.pi)))

        plot_data=np.array(plot_data)

        np.save(path+label_pr, np.column_stack((t*3.3e-5,plot_data)))

        plt.plot(t*3.3e-5,plot_data)
        plt.xlabel("t,s")
        plt.ylabel("Total "+label_pr)
        plt.savefig('plots/integ_plot_'+label_pr+'.png', bbox_inches='tight',dpi=300)
        print('plots/integ_plot_'+label_pr+'.png')
        plt.clf()
        plt.close()


integrated_plot('p', path="plots/res26_1HR/", kinetic_component='all', relative=False)






def vel_plot( remove_avg_omega:bool=False):
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


    x_plot=phi/(np.sqrt(2)) #projection
    y_plot=np.sqrt(2)*np.sin(theta)

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
                    x_plot_full[face_num][i]+=2*np.pi/np.sqrt(2)
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

    ax[0].streamplot(X_gr,Y_gr,xd_gr,yd_gr,color=v,norm=norm2, cmap=colorm2, arrowsize=2,density = 1.7)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colorm),cax=ax[1], orientation='horizontal', label=r'$\Sigma$, $10^7 \rm g \ \rm cm^{-2}$ ')
    #fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colorm),cax=ax[1], orientation='horizontal', label=r'Speed, c ')
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap=colorm2),cax=ax[2], orientation='horizontal', label=r"v/c")
    fig.savefig('plots/vel_plot.png', bbox_inches='tight',dpi=150)
    plt.clf()
    plt.close()



def butterfly_diagram(value:str, path:str='results/', min:float=None, max:float=None, skipstep:int=1, log:bool=False, skiprows:int=0, total_steps:int=0, tilt_angle:float=0, n_bins:int=100): 
    if(total_steps==0):
        total_steps=int(1e10)

    gam=1.25

    face_centers=pd.read_table(path+'face_centers.dat', header=None, delimiter=r"\s+")
    face_centers=np.array(face_centers)/(np.array([np.linalg.norm(np.array(face_centers), axis=1),
        np.linalg.norm(np.array(face_centers), axis=1),np.linalg.norm(np.array(face_centers), axis=1)]).T)

    # Latitude analog: -pi/2 to pi/2
    theta_fc = -np.acos(face_centers[:,1] * np.sin(tilt_angle) + face_centers[:,2] * np.cos(tilt_angle)) + np.pi/2

    if(value=='rho'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr=r'$\Sigma$, $10^7 \rm g \ \rm cm^{-2}$ '
    elif(value=='p'):
        data_rho=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr='Pressure'
    elif(value=='omega_z'):
        data_rho=pd.read_table(path+'omega.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr='Omega_z'
    elif(value=='omega'):
        data_rho=pd.read_table(path+'omega.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+", skiprows=skiprows)

        maxstep=np.min([len(data_rho.loc[:,0]), total_steps])

        for i in range(0,maxstep,skipstep):
            rho=data_rho.loc[i,1:]
            #L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T
            omega=(data_Ly.loc[i,1:] * np.sin(tilt_angle) + data_Lz.loc[i,1:] * np.cos(tilt_angle))/(data_rho.loc[i,1:]*np.cos(theta_fc)**2)
            data_rho.loc[i,1:]=omega
        label_pr='Omega'
    elif(value=='vort'):
        data_rho=pd.read_table(path+'curl.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr=r'Vorticity, $\Omega$ '
    elif(value=='beta'):
        data_rho=pd.read_table(path+'beta.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr=r'$\beta$ '
    elif(value=='mach'):
        data_rho=pd.read_table(path+'mach.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr='Mach number'
    elif(value=='c_s'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr='Speed of sound'
        data_rho.loc[:,1:]=data_p.loc[:,1:]/data_rho.loc[:,1:]
        data_rho.loc[:,1:]=np.sqrt(gam*data_rho.loc[:,1:])
    elif(value=='entropy'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_beta=pd.read_table(path+'beta.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        label_pr='Entropy'
        data_rho.loc[:,1:]=data_p.loc[:,1:]/(data_rho.loc[:,1:]**  ( (10-3*data_beta.loc[:,1:])/(8-3*data_beta.loc[:,1:]) )   )
    elif(value=='vel_abs'):
        label_pr='Speed'
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+", skiprows=skiprows)

        maxstep=np.min([len(data_rho.loc[:,0]), total_steps])

        for i in range(0,maxstep,skipstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            data_rho.loc[i,1:]=np.linalg.norm(np.cross(face_centers,L), axis=1)/rho0
    elif(value=='L'):
        label_pr=r'T (K)'
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")

        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")

        data_beta=pd.read_table(path+'beta.dat', header=None, delimiter=r"\s+")

        #maxstep=np.min([len(data_rho.loc[:,0]), len(data_p.loc[:,0]),len(data_Lx.loc[:,0]),len(data_Ly.loc[:,0]),len(data_Lz.loc[:,0]), len(data_beta.loc[:,0])])
        maxstep=np.min([len(data_rho.loc[:,0]), total_steps])


        data_beta=np.array(data_beta.loc[:,1:])
        vel=[]

        for i in tqdm(range(0,maxstep,skipstep)):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            vel.append(np.linalg.norm(np.cross(face_centers,L), axis=1)/rho0)
        vel=np.array(vel)


        surface_area=4*np.pi/len(face_centers)
        kappa = 3.4e6 #// scattering opacity in 1/Sigma_unit (R_unit^2/M_unit)
        GM = 0.217909 # grav parameter in R_unit^3/t_unit^2
        g_eff = - vel * vel + GM
        g_eff[g_eff<0]=0
        j=0
        for step in tqdm(range(0,maxstep,skipstep)):
            data_rho.loc[step,1:]=np.power((g_eff[j] / kappa * (1 - data_beta[step])* surface_area)*9e27*1e12/(3.3e-5) /(5.6e-5),1/4) 
            j+=1
    else:
        print("wrong type of plot value")
        return

    maxstep=int(np.min([len(data_rho.loc[:,0]), total_steps]))
    
    if(log):
        data_rho.loc[:,1:]=np.log10(data_rho.loc[:,1:].astype(float))
        label_pr=r'$log_{10}$ of '+label_pr

    t_full = data_rho.iloc[:,0].to_numpy(dtype=float) * 3.3e-5
    
    bins = np.linspace(np.min(theta_fc), np.max(theta_fc), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    digitized = np.digitize(theta_fc, bins)
    
    # Pre-calculate time steps to process
    steps = list(range(0, maxstep, skipstep))
    t_plot = t_full[steps]
    Z = np.zeros((len(steps), n_bins))
    
    for idx, i in enumerate(tqdm(steps)):
        row_data = data_rho.iloc[i, 1:].to_numpy(dtype=float)
        for b in range(1, n_bins + 1):
            mask = (digitized == b)
            if np.any(mask):
                Z[idx, b-1] = np.mean(row_data[mask])
            else:
                Z[idx, b-1] = np.nan
                
    T_mesh, Theta_mesh = np.meshgrid(t_plot, bin_centers)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if min is None:
        min_val = np.nanmin(Z)
    else:
        min_val = min
        
    if max is None:
        max_val = np.nanmax(Z)
    else:
        max_val = max
        
    mesh = ax.pcolormesh(T_mesh, Theta_mesh, Z.T, cmap='viridis', vmin=min_val, vmax=max_val, shading='auto')
    
    ax.set_xlabel('Time [s]', fontsize=20)
    ax.set_ylabel(r'Latitude $\theta$ [rad]', fontsize=20)
    ax.set_title(f'Butterfly Diagram - {label_pr}', fontsize=22)
    ax.set_ylim([-np.pi/2, np.pi/2])

    data_ts=pd.read_table(path+'lightcurve0.dat', header=None, delimiter=r"\s+", skiprows=skiprows)
    lavg=(data_ts.loc[:,1]-np.mean(data_ts.loc[:,1]))/np.max(data_ts.loc[:,1]-np.mean(data_ts.loc[:,1]))
    ax.plot(data_ts.loc[:,0]*3.3e-5, lavg, color='tab:red', alpha=0.7, label='Lightcurve (scaled)')
    ax.legend(loc='upper right', fontsize=20, framealpha=0.5)

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(label_pr, fontsize=20)
    
    plt.tight_layout()
    fig.savefig(f'plots/butterfly_{value}.png', bbox_inches='tight', dpi=200)
    plt.clf()
    plt.close()


butterfly_diagram('rho', path='plots/res26_3alpha2e4/', min=None, max=None, skipstep=100, log=True, skiprows=0, total_steps=0, tilt_angle=0.1, n_bins=100)


def plot_vs_theta(value:str,path:str='results/', skipstep:int=1, ylim_min:float=0, ylim_max:float=0, binning:bool=False, n_bins:int=50):


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

        face_centers=np.array(face_centers)/(np.array([np.linalg.norm(np.array(face_centers), axis=1),
        np.linalg.norm(np.array(face_centers), axis=1),np.linalg.norm(np.array(face_centers), axis=1)]).T)

        for i in range(maxstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_rho.loc[i,1:]
            data_rho.loc[i,1:]=np.linalg.norm(np.cross(face_centers,L), axis=1)/rho0
    else:
        print("wrong type of plot value")
        return

    maxstep=len(data_rho.loc[:,0])

    if(ylim_min==0 and ylim_max==0):
        ylim_min=np.min(data_rho.loc[:,1:])*0.9
        ylim_max=np.max(data_rho.loc[:,1:])*1.1

    for i in tqdm(range(maxstep)):
        if((i % skipstep)==0 ):

            if(binning):
                bins=np.linspace(-np.pi/2, np.pi/2, n_bins)
                digitized = np.digitize(theta_fc, bins)
                bin_means=[np.mean(data_rho.loc[i,1:][digitized==j]) for j in range(1, len(bins))]
                bin_centers=0.5*(bins[:-1]+bins[1:])
                plt.plot(bin_centers, bin_means)
            else:
                plt.scatter(theta_fc, data_rho.loc[i,1:], s=2)
            plt.xlabel(r'$\theta$')
            plt.ylabel(label_pr)
            plt.title('t='+"{:.4f}".format(data_rho.loc[i,0]*3.3e-5)+' s')
            plt.ylim([ylim_min, ylim_max])
            plt.savefig('plots/plot_vs_theta_'+"{0:0>4}".format(i)+'.png', bbox_inches='tight',dpi=200)
            plt.clf()
            plt.close()



plot_vs_theta('entropy',path="plots/res26_3alpha2e4/", skipstep=300, ylim_min=0, ylim_max=0, binning=True, n_bins=50)