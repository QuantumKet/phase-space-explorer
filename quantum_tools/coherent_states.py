def theta_QP(Q:float,P:float):
    """
    la correspondencia entre QP y el ángulo theta
    """
    return math.acos(1-((Q**2)+(P**2))/2)
    
def phi_QP(Q:float,P:float):
    """
    la correspondencia entre QP y el ángulo phi
    """
    return math.fmod(math.atan(-P/Q),2*math.pi)
    
def atomic_coherent_state(J:int,Q:float,P:float):
    """
    La obtención del estado coherente atómico z(Q,P) de dimensión 2J+1
    """
    theta = theta_QP(Q,P)
    phi = phi_QP(Q,P)
    if theta == 0 or Q == 0:
        return np.zeros(2*J+1)
    coherent_state = []
    for M in range(-J,J+1):
        #print(M+J)
        c_m = 0.5*math.log(comb(2*J,J+M,exact=True))+math.log(math.sin(0.5*theta))*(J-M)+math.log(math.cos(0.5*theta))*(J+M)+-1j*(J-M)*phi
        coherent_state.append(c_m)
    return np.exp(coherent_state)[::-1]
    
def basis_overlap(eigenvectors,coherent_state):
    """
    El overlap entre un estado coherente y la eigenbase
    """
    projections = []
    for vec in eigenvectors:
        basis_vec = list(map(lambda x:x[0],vec.full()))[::-1]
        #print(len(basis_vec),len(coherent_state))
        #basis_vec = vec.trans()[0]
        projections.append(np.dot(basis_vec,coherent_state))
    return projections
