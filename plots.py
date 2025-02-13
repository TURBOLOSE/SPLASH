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


def projection_plots(value, print_residuals:bool=False, print_log:bool=False, add_streamplot:bool=False): 
    #value = rho,p,omega
    skipstep=1
    
    gam=1.25

    #path='results/'
    #path='plots/big_quad_next/'
    path='plots/new split test/2 layers/'
    #path='plots/shock_test/'

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
    elif(value=='c_s'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_p=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        label_pr='Speed of sound'
        data_rho.loc[:,1:]=data_p.loc[:,1:]/data_rho.loc[:,1:]
        data_rho.loc[:,1:]=np.sqrt(1.25*data_rho.loc[:,1:])
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

    if(print_residuals):
        for i in range(1,maxstep):
            data_rho.loc[i,:]-=data_rho.loc[0,:]
            data_rho.loc[i,1:]/=data_rho.loc[0,1:]
        data_rho.loc[0,:]-=data_rho.loc[0,:]
        label_pr+=" residuals"

    if(print_log):
        data_rho.loc[:,1:]=np.log10(data_rho.loc[:,1:])


    if(add_streamplot):
        data_dens=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        data_Lx=pd.read_table(path+'Lx.dat', header=None, delimiter=r"\s+")
        data_Ly=pd.read_table(path+'Ly.dat', header=None, delimiter=r"\s+")
        data_Lz=pd.read_table(path+'Lz.dat', header=None, delimiter=r"\s+")
        face_centers=pd.read_table(path+'face_centers.dat', header=None, delimiter=r"\s+")
        face_centers=np.array(face_centers)/(np.array([np.linalg.norm(np.array(face_centers), axis=1),
        np.linalg.norm(np.array(face_centers), axis=1),np.linalg.norm(np.array(face_centers), axis=1)]).T)

        maxstep=len(data_dens.loc[:,0])
        vel=[]
        for i in range(maxstep):
            L=np.array([data_Lx.loc[i,1:],data_Ly.loc[i,1:],data_Lz.loc[i,1:]]).T 
            rho0=data_dens.loc[i,1:]
            vel.append(-np.cross(face_centers,L)/np.array([rho0,rho0,rho0]).T)
        del data_dens #deallocating useless memory
        del data_Lx
        del data_Ly
        del data_Lz
        gc.collect()

        vel=np.array(vel)
        theta_fc=np.arccos(face_centers[:,2])
        phi_fc=np.arctan2(face_centers[:,1],face_centers[:,0])
        x_fc=phi_fc/np.sqrt(2)
        y_fc=np.sin(-theta_fc+np.pi/2)*np.sqrt(2)

        yd=np.sqrt(2)*np.cos(theta_fc)*(np.cos(theta_fc)*np.cos(phi_fc)*vel[:,:,0]+np.cos(theta_fc)*np.sin(phi_fc)*vel[:,:,1]-np.sin(theta_fc)*vel[:,:,2])
        xd=1/np.sqrt(2)*((-np.sin(phi_fc)*vel[:,:,0]+np.cos(phi_fc)*vel[:,:,1])/np.sin(theta_fc))

        X_gr, Y_gr=np.meshgrid(np.linspace(-2.2,2.2, 100),np.linspace(-1.4, 1.4, 100))

        xd_gr=[]
        yd_gr=[]
        for i in range(maxstep):
            mask=np.logical_or(np.isnan(xd[i], where=False),np.isnan(xd[i], where=False))
            xd_gr.append(griddata(np.stack([x_fc[mask].T, y_fc[mask].T]).T, xd[i][mask],(X_gr,Y_gr), method='nearest'))
            yd_gr.append(griddata(np.stack([x_fc[mask].T, y_fc[mask].T]).T, yd[i][mask],(X_gr,Y_gr), method='nearest'))    

        xd_gr=np.array(xd_gr)
        yd_gr=np.array(yd_gr)

        colorm2 = plt.get_cmap('inferno')
        v=np.sqrt(xd_gr**2+yd_gr**2)
        norm2 = mpl.colors.Normalize(vmin=np.min(v), vmax=np.max(v))




    data_faces=pd.read_table(path+'faces.dat', header=None, delimiter=r"\s+", names=['col' + str(x) for x in range(6) ])
    face_centers=pd.read_table(path+'face_centers.dat', header=None, delimiter=r"\s+")

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


    for face_num,face in enumerate(faces): #fix x
        sign_arr=np.sign(x_plot_full[face_num])
        if( (not (0 in sign_arr)) and (1 in sign_arr) and (-1 in sign_arr) and (np.min(np.abs(x_plot_full[face_num])) > 1)):
            for i,element in enumerate(x_plot_full[face_num]):
                if(element < 0):
                    x_plot_full[face_num][i]+=2*np.pi/np.sqrt(2)


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
    min_rho=np.min( data_rho.loc[:maxstep,1:len(x_plot)])
    max_rho=np.max( data_rho.loc[:maxstep,1:len(x_plot)])
    #min_rho=0
    #max_rho=2e-4
    #max_rho=1.8

    #data_rho.loc[:,1:]*=2
    #max_rho=50

    norm = mpl.colors.Normalize(vmin=min_rho, vmax=max_rho)
    mpl.rcParams.update({'font.size': 25})

   


    for i in tqdm(range(maxstep)): #dens
        if((i % skipstep)==0 ):

            collection = PatchCollection(patches)
            colors=colorm(norm(data_rho.loc[i,1:len(faces)]))

            fig, ax = plt.subplots(figsize=(16, 10), layout='constrained', nrows=2,height_ratios=[15,1])
            #fig.tight_layout()
            plt.subplots_adjust(hspace=10)
            #rho=(np.array(data_rho.loc[i,1:len(faces)])-min_rho)/(max_rho-min_rho)
            fig.suptitle('t='+"{:.4f}".format(data_rho.loc[i,0]*3.3e-5)+' s')
            ax[0].set_xlabel(r'$\varphi / \sqrt{2}$', fontsize=25)
            ax[0].set_ylabel(r'$\sqrt{2}  \cos(\theta )$', fontsize=25)

            #collection = PatchCollection(patches)
            ax[0].add_collection(collection)
            collection.set_color(colors)
            ax[0].set_xlim([-2.5, 3.4])
            ax[0].set_ylim([-1.5, 1.5])

            if(add_streamplot):
                ax[0].streamplot(X_gr,Y_gr,xd_gr[i],yd_gr[i],color=v[i],norm=norm2, cmap=colorm2, arrowsize=3)


            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colorm),cax=ax[1], orientation='horizontal', label=label_pr)
            fig.savefig('plots/fig'+"{0:0>4}".format(i)+'.png', bbox_inches='tight',dpi=300)
            plt.clf()
            plt.close()
    



#projection_plots("c_s", print_residuals=False, print_log=False, add_streamplot=False)
projection_plots('vort', print_residuals=False, print_log=False, add_streamplot=False)



def integrated_plot(value): 
    #value = rho,p


    path='results/'


    if(value=='rho'):
        data_rho=pd.read_table(path+'rho.dat', header=None, delimiter=r"\s+")
        label_pr=r'm, $10^7 \rm g $ '
    elif(value=='p'):
        data_rho=pd.read_table(path+'p.dat', header=None, delimiter=r"\s+")
        label_pr='Pressure'
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
        plot_data.append(np.sum(np.array(data_rho.loc[step,1:])*surface_areas))

    plot_data=np.array(plot_data)

    plt.plot(t*3.3e-5,plot_data)
    plt.xlabel("t,s")
    plt.ylabel("total "+label_pr)
    plt.savefig('plots/integ_plt.png', bbox_inches='tight',dpi=300)
    plt.clf()
    plt.close()


integrated_plot('rho')






def vel_plot():
    gam=1.25

    #turn_angle=np.pi/2

    turn_angle=0
    #path_to_res='results/'
    path_to_res='plots/big_quad_next/'
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


    #phi_d-=np.average(phi_d)


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
    
    fig.suptitle('t='+"{:10.4f}".format(0.7920)+" s")
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



vel_plot()

