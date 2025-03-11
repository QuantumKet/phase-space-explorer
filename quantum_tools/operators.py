import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import cmath

def floquet_matrix(J:int,alpha:float=0.84,k:float=0.5):
    """
    El operador de Floquet para el trompo pateado
    """
    Jx = qt.jmat(J,'x')
    Jz = qt.jmat(J,'z')
    e_z = -1j*alpha*Jz
    e_xx = ((-1j*k)/(2*J))*(Jx*Jx) 
    F = e_z.expm()*e_xx.expm()
    return F

def parity_separator(F,J):
    """
    Una función que separa la matriz de Floquet en paridad
    """

    matrix_pos_gen = np.array([complex(F[[i],[j]]) for i,j in product(range(0,2*J+1,2),range(0,2*J+1,2))]).reshape(-1,J+1) 
    matrix_neg_gen = np.array([complex(F[[i],[j]]) for i,j in product(range(1,2*J+1,2),range(1,2*J+1,2))]).reshape(-1,J)
    return qt.Qobj(matrix_pos_gen),qt.Qobj(matrix_neg_gen)

def r_prom(cuasienergias):
    """
    La diferencia entre niveles de energía establecida por Oganesyan y Huse
    """
    ri_ls = []
    for i in range(len(cuasienergias)-2):
        ri_ls.append(r_p(cuasienergias,i))
    return sum(ri_ls)/(len(ri_ls))
        
def parity_operator(J:int):
    """
    El operador de paridad aplicado al QKT
    """
    parity_op = (-1j*np.pi)*qt.jmat(J,'z')
    return qt.jmat(J,'z').expm()

def eigensort(eigenvals,eigenvecs):
    """
    El ordenamiento de la eigenbase en términos de las fases de las quasienergias
    """
    eigensorted = []
    for eigv, eigvec in zip(eigenvals,eigenvecs):
        eigensorted.append((cmath.phase(eigv),eigvec))
    eigensorted.sort(key=lambda x:x[0])
    return eigensorted

def r_graf(patadas,J:int=100, parity:str='POS'):
    """
    La obtención de la estadística de niveles por diferencia de primeros vecinos
    Esta función obtiene <r>, en un rango o lista de patadas y una dimensión 2J+1
    """
    plt.ylim(0.0,1.0)
    plt.title(fr'<r> $\alpha=0.84$, J={J}')
    plt.xlabel('k')
    plt.ylabel('<r>')
    plt.axhline(y = 0.39, color = 'r', linestyle = '--') #El límite inferior
    plt.axhline(y = 0.5, color = 'r', linestyle = '--') #El límite superior
    r_values = [] #Aquí guardaremos la lista de <r>, pero podemos graficar cada punto con una gráfica scatter
    for i in patadas:
        F = Floquet_Matrix(J,k=i)
        F_ppar,F_npar=Floquet_Parity_Matrix(F,J)
        #Nos quedamos solo con eigenenergias pares
        if parity == 'POS':
            eigenenergie, eigenstate = F_ppar.eigenstates()
        else:
            eigenenergie, eigenstate = F_npar.eigenstates()
        phases = [cmath.phase(x) for x in eigenenergie] #Saca la fase entre -pi y pi OBS: ESTO NO SEPARA EN PARIDAD
        phases.sort() #Ordena de menor a mayor
        r = r_prom(phases)
        print(f'El valor promedio de r con J= {J} y k= {i} es {r}')
        r_values.append(r)
    plt.plot(patadas,r_values,'b.-')