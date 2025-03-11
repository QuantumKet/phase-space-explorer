import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

def kicked_top_map(alpha:float=0.84,coord:list=[1,1,1], k:float=0.5):
    """
    El mapeo estroboscopio del QKT
    """
    x, y, z = coord
    new_x = (x * np.cos(alpha)) - (y * np.sin(alpha) * np.cos(k * x)) + (z * np.sin(alpha) *np.sin(k * x))
    new_y = (x*np.sin(alpha)) + (y * np.cos(alpha) * np.cos(k * x)) - (z * np.cos(alpha) * np.sin(k * x))
    new_z = (y*np.sin(k*x)) + (z*np.cos(k*x))
    return new_x, new_y, new_z

def poincare_gen(alpha:float=0.84,k:float=0.5,n:int=500):
    """
    Esta función traza trayectorias sobre la esfera de bloch usando el mapeo estroboscopico
    """
    # Generar condiciones iniciales aleatorias en la superficie de la esfera
    phi = np.random.uniform(0, 2 * np.pi, n)
    cos_theta = np.random.uniform(-1, 1, n)
    theta = np.arccos(cos_theta)
    initial_coords = np.column_stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), cos_theta))

    # Crear una figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Configurar la esfera
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='c', alpha=0.1, antialiased=False)
    #legend_elements = []
    # Iterar para cada condición inicial
    for i in range(n):
        color = plt.cm.jet(i / n)  # Color basado en mapa de colores 'jet'
        coord = initial_coords[i]
        
        x_values = [coord[0]]
        y_values = [coord[1]]
        z_values = [coord[2]]
        
        for _ in range(2000):  # Número de iteraciones del mapeo por condición inicial
            coord = kicked_top_map(alpha,coord, k)
            x_values.append(coord[1])
            y_values.append(coord[2])
            z_values.append(coord[0])
        ax.scatter(x_values, y_values, z_values, color=color, s=1 )  # Ajustar el grosor aquí
        #legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Condición {i+1}', markerfacecolor=color, markersize=5))


    # Configurar etiquetas
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trayectorias bajo el Mapeo F - Kicked Top')
    #ax.legend(handles=legend_elements, loc='upper left')
    # Mostrar el gráfico
    #plt.show()
    return plt

def stereographic_projection(coord:list):
    """
    La proyección de la esfera al disco de radio 2
    """
    x,y,z = coord
    Q = np.sqrt(2)*x / np.sqrt(1-z)
    P = np.sqrt(2)*y / np.sqrt(1-z)
    return Q,P
     
def sphere_projection(Q:float,P:float):
    """
    La proyección del disco a la esfera
    """
    Sx = Q*np.sqrt(1- ((Q**2 + P**2)/4))
    Sy = P*np.sqrt(1- ((Q**2 + P**2)/4))
    Sz = ((Q**2 + P**2)/2)-1
    return Sx,Sy,Sz

def disk_limits_to_sphere_limits(x_min:float, x_max:float, y_min:float, y_max:float):
    # Convert disk limits to sphere limits using inverse stereographic projection
    """
    Una traducción de los límites de la sección de Poincaré para la esfera de Bloch
    """
    phi_min = np.arctan2(y_min, x_min)
    phi_max = np.arctan2(y_max, x_max)
    theta_min = 2 * np.arctan(1 / np.sqrt(x_min**2 + y_min**2))
    theta_max = 2 * np.arctan(1 / np.sqrt(x_max**2 + y_max**2))
    return phi_min, phi_max, theta_min, theta_max

#Primero definiremos un modo de elegir las coordenadas a colorear y poblaremos el resto del espacio con 1000 condiciones
def poincare_section(alpha:float=0.84,k:float=0.5,points:list=[],n:int=1000,section='full',k_n:int=250,k_n_p:int=500,uid:str=''):
    """
    La sección de Poincaré del sistema
    """
    np.random.seed(seed=1) #Esto fija la aleatoreidad
    if isinstance(section,list) and len(section)==2:
        q_min,q_max = section[0]
        p_min,p_max = section[1]
        print(f'SHOWING ({q_min},{q_max})x({p_min},{p_max}) MAP ZONE')
        phi_min,phi_max,theta_min, theta_max = disk_limits_to_sphere_limits(q_min,q_max,p_min,p_max)
        phi = np.random.uniform(phi_min, phi_max, n)
        cos_theta = np.random.uniform(np.cos(theta_max), np.cos(theta_min), n)
    else:
        print('FULL MAP')
        q_min,q_max = (-2,2)
        p_min,p_max = (-2,2)
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_theta = np.random.uniform(-1, 1, n)
    theta = np.arccos(cos_theta)
    initial_coords = np.column_stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), cos_theta))
    plt.figure(figsize=(20,20),clear=True)
    plt.clf()
    #plt.rcParams["figure.figsize"] = (20,20)
    plt.title(fr'$\alpha$={alpha}, $k$={k}, $n$={n}',fontsize=40)
    plt.grid(color='0.9', linestyle='-', linewidth=1)
    plt.axis('square')
    plt.xlim(q_min,q_max)
    plt.ylim(p_min,p_max)
    plt.xlabel('Q',fontsize=30)
    plt.ylabel('P',fontsize=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    for i in range(n):
        coord = initial_coords[i]
        Q,P = stereographic_projection(coord)
        Q_values = [Q]
        P_values = [P]
        for _ in range(k_n):  # Número de iteraciones del mapeo por condición inicial
            coord = kicked_top_map(alpha,coord, k)
            Q,P = stereographic_projection(coord)
            Q_values.append(Q)
            P_values.append(P)
        plt.scatter(Q_values,P_values,color='0.8',s=.1)
    for i,point in enumerate(points):
        coord = sphere_projection(point[0],point[1])
        try:
            color = point[2]
        except:
            color = plt.cm.jet(i / len(points))
        Q,P = stereographic_projection(coord)
        Q_values = [Q]
        P_values = [P]
        for _ in range(k_n_p):
            #print(coord)
            coord = kicked_top_map(alpha,coord, k)
            Q,P = stereographic_projection(coord)
            Q_values.append(Q)
            P_values.append(P)
        
        #print(Q_values,P_values)
        plt.scatter(point[0],point[1],color=color,s=40,label=f'(Q,P)=({point[0]},{point[1]})')
        plt.scatter(Q_values,P_values,color=color,s=13)
        if len(points)<50:
            plt.legend()
    if len(uid)>0:
        plt.savefig(f'./POINCARESECTIONS/KICKEDTOP_a={alpha}_k={k}_n={n}_{uid}.png')
    else:
        plt.savefig(f'./POINCARESECTIONS/KICKEDTOP_a={alpha}_k={k}_n={n}.png')
    plt.show()
    #plt.close()
    return plt