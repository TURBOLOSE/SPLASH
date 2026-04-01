import numpy as np
import pandas as pd
from scipy.integrate import odeint
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
    p_0=1
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
        if(r[face_num]<R0):
            # Smoothly taper the velocity to zero using a cosine bell
            smooth_factor = 0.5 * (1 - np.cos(np.pi * r[face_num] / R0))
            local_omega = np.array([omega_cyclone * smooth_factor,0,0])

            l.append(rho[face_num]*np.cross(R,np.cross(local_omega,R))/(np.linalg.norm(R)**2))
            v.append(np.cross(local_omega,R)/np.linalg.norm(R))
            
            rho[face_num]*=1.2
            dp = 0.5 * rho[face_num] * (np.linalg.norm(v[face_num])**2) * (1 - np.cos(np.pi * r[face_num] / R0))
            p[face_num] += dp

        else:
            l.append(np.array([0,0,0]))
            v.append(np.array([0,0,0]))

        # if(theta[face_num]<np.pi/2+0.5+0.3 and theta[face_num]>np.pi/2+0.5-0.3):
        #     l.append(rho[face_num]*np.cross(R,np.cross(omega,R))/(np.linalg.norm(R)**2))
        #     v.append(np.cross(omega,R)/np.linalg.norm(R))
        # else:
        #     l.append(np.array([0,0,0]))
        #     v.append(np.array([0,0,0]))
        #l.append(np.array([0,0,0]))
        #v.append(np.array([0,0,0]))

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
    sol = odeint(press, p_0, theta_l, hmax=0.01)

    if sol.ndim == 1:
        p_short = sol
    else:
        p_short = sol[:, 0]

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
make_input_5_coriolis()
#make_input_5_const_entr()
#make_input_5_sp_layer()
#make_input_5_sp_layer_exp()
#make_input_5_sp_layer_diff_rot()
#make_input_4()
#make_input_6_cos_bell()




