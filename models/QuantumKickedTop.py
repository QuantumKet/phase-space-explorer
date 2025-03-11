"""
Author: Ariel Galindo
Aquí definimos el trompo pateado cuántico con algunas de sus características más importantes
Sin embargo, en los demás scripts se encuentra una gran variedad de herramientas que sirven para
operar este modelo.
"""
from quantum_tools.operators import floquet_matrix, r_prom, eigensort, parity_separator
from classic_tools.phase_space import poincare_section
from quantum_tools.trajectory_classificator import *
from quantum_tools.coherent_states import atomic_coherent_state, basis_overlap

class QKT():
    def __init__(self, J:int, alpha:float=0.84,k:float=0.5):
        self.J = J
        self.alpha = alpha
        self.k = k
        # Definiciones adicionales
        F = floquet_matrix(J,alpha,k)
        eigenvalues, eigenvectors = F.eigenstates()
        eigensorted = eigensort(eigenvalues,eigenvectors)
        #Propiedades
        self.floquet_matrix = F
        self.eigenbase = [val[0] for val in eigensorted]
        self.quasienergies = [val[1] for val in eigensorted]
    
    def level_statistics(self,parity='POS'):
        F_ppar, F_npar = parity_separator(self.floquet_matrix,self.J)

        if parity == 'POS':
            eigenvalues, eigenvectors = F_ppar.eigenstates()
        else:
            eigenvalues, eigenvectors = F_npar.eigenstates()

        eigensorted = eigensort(eigenvalues,eigenvectors)

        return r_prom([val[1] for val in eigensorted])

    def ldos(self,point):
        from numpy import abs as npabs
        coh = atomic_coherent_state(J,point[0],point[1]) #Este es el 6 ciclo
        return npabs(basis_overlap(eigenvectors,coh))**2
    
    def pr(self,point):
        from numpy import abs,sum as npabs,npsum
        coh = atomic_coherent_state(J,point[0],point[1]) #Este es el 6 ciclo
        ldos = npabs(basis_overlap(eigenvectors,coh))**2
        return 1/npsum(ldos**2)

    #Pseudo classic dynamics

    def classis_phase_space(self,points:list=[],iterations=250,uid=''):
        poincare_section(k=self.k,points=points, k_n_p=iterations,uid=uid)

    def phase_space_exploration(self,points,show_space_phase=False,show_data=True,detection_mode='ALL',data_route='./DATA/automaticexplorer_data.csv'):
        """
        El explorador del espacio fase
        """
        k = self.k
        alpha =  self.alpha
        J = self.J
        
        quasienergies = self.quasienergies
        eigenvectors = self.eigenbase
        
        periodic_coeff,noisy_coeff,random_coeff = calculate_space_coeffs(len(eigenvectors))
        #print(f"Dimensión del espacio:{len(eigenvectors)}\nCoeficientes del espacio:\nCoeficiente periodico:{periodic_coeff}")
        #print(f"Coeficiente ruidoso: {noisy_coeff}\nCoeficiente aleatorio: {random_coeff}")
        data = (J,alpha,k,periodic_coeff,noisy_coeff,random_coeff)
        with open(data_route,'a') as data_file:
                writer_object = writer(data_file)
                writer_object.writerow(("J","alpha","k","periodic_coeff","noisy_coeff","random_coeff"))
                writer_object.writerow(data)
                writer_object.writerow(("(Q,P)","FVP","C","PR","p_type","period"))
                data_file.close()

        for point in points:
            if show_space_phase:
                poincare_section(k=k,points=[point])
            coh = atomic_coherent_state(J,point[0],point[1]) #Este es el 6 ciclo
            ldos = np.abs(basis_overlap(eigenvectors,coh))**2
            smoothed_amps=convolved_smoothing(ldos,121)
            smoothed_dos_amps=convolved_smoothing(smoothed_amps,121)
            smoothed_tres_amps=convolved_smoothing(smoothed_dos_amps,121)
            
            sup_peaks, _ = find_peaks(smoothed_tres_amps,height=np.median(smoothed_tres_amps))
            low_peaks, _ = find_peaks(-smoothed_tres_amps)
            fvp = prom_visibility(smoothed_tres_amps,(sup_peaks,low_peaks))
            
            r_x = autocorrelation(smoothed_tres_amps)
            period = estimate_period(r_x)
            periodicity_energy = periodicity_coefficient_energy(r_x)
            pr = 1/np.sum(ldos**2)
            pr = pr/len(ldos)
            ptype = ''
            if fvp >=0.1:
                if periodicity_energy <= noisy_coeff and pr<0.1:
                    #print('Trayectoria estacionaria detectada')
                    ptype = 'E'
                elif (periodicity_energy>noisy_coeff or periodicity_energy>=periodic_coeff) and pr<0.1:
                    #print('Ciclo detectado')
                    #print(f'PERIODO ESTIMADO: {period}')
                    ptype='kC'
                elif periodicity_energy>=noisy_coeff and pr>=0.1:
                    #print('Trayectoria pegajosa detectada')
                    ptype='P'
                else:
                    #print(f'({point[0]},{point[1]}) no posible de clasificar pero con visibilidad alta')
                    #print(fvp,periodicity_energy,pr)
                    continue
            elif fvp<0.1:
                
                if periodicity_energy<=noisy_coeff and pr>=0.1:
                    #print('Trayectoria caótica detectada')
                    ptype='C'
                elif periodicity_energy <= noisy_coeff and pr<0.1:
                    #print('Trayectoria estacionaria detectada')
                    ptype = 'E'
                elif periodicity_energy>=noisy_coeff and pr>=0.1:
                    #print('Trayectoria pegajosa detectada')
                    ptype='P'
                else:
                    #print(f'({point[0]},{point[1]}) no posible de clasificar pero con visibilidad baja')
                    continue
            else:
                #print(f'({point[0]},{point[1]}) no posible de clasificar pero con visibilidad baja')
                continue
            if ptype != 'kC':
                period = 'N/A'
            if show_data:
                print(f"(Q,P)=({point[0]},{point[1]})")
                #print(f"Periodo estimado: {period}")
                print(f"Coeficiente de periodicidad: {periodicity_energy}")
                print(f"FVP: {fvp}")
                print('PR= ',pr)
                
                fig, axs = plt.subplots(3, 1, figsize=(15, 10))
        
                axs[0].plot(quasienergies,ldos, ls='--',marker='o',label=f'({point[0]},{point[1]})')
                axs[0].set_xlabel(r'$\phi$')
                axs[0].set_ylabel('$Cm^2$')
                axs[0].set_title('LDOS')
                axs[0].legend()
                axs[0].grid()
        
                axs[1].plot(quasienergies,smoothed_tres_amps, ls='--',marker='o',label=f'({point[0]},{point[1]})')
                axs[1].set_xlabel(r'$\phi$')
                axs[1].set_ylabel('$Cm^2$')
                axs[1].set_title('Suavizado')
                axs[1].legend()
                axs[1].grid()
        
                axs[2].stem(r_x)
                axs[2].set_xlabel('Retardo')
                axs[2].set_ylabel('Autocorrelación')
                axs[2].set_title('Función de Autocorrelación')
                axs[2].grid()
        
                plt.tight_layout()
                plt.savefig(f'./AUTOCORRELATION/(Q,P)=({point[0]},{point[1]})_J={J}_k={k}.jpg')
                plt.show()
                plt.close()           
            data = (f"({point[0]},{point[1]})",fvp,periodicity_energy,pr,ptype,period)
            with open(data_route,'a') as data_file:
                writer_object = writer(data_file)
                writer_object.writerow(data)
                data_file.close()
            
        return 'Exploración finalizada'






        