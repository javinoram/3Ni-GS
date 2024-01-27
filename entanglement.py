from base import *

"""
Lectura de parametros de la linea de comandos, si es 3D y 1D, y la configuracion del acople
"""
structure = sys.argv[1]
conf = sys.argv[2]
if structure == '3D':
      j1,j2,j3= J3D
else:
      j1,j2,j3= J1D


"""
Funcion que resume el workflow del calculo de la traza parcial
"""
def workflow(j1, j2, j3, j, hz, hx, conf):
        H = hamiltonian([j1, j2, j3, j, hz, hx, int(conf)])
        _, vv= Numpyget_eigen(H)

        state = np.array( [[j] for j in vv[:,0]] )
        state_dag = state.conj().T
        density = state.dot( state_dag )
        ee = la.eigvalsh( NumpyPartialTraceLR(np.real(density), 3, 1) )
        return ee[ee>0]
       

"""
Variables del sistema, campo magnetico en Z, X y el exchange entre moleculas
"""
Hz = np.linspace(1e-3, 10, 301)
Hx = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
exchanges = np.linspace(-0.6, 0.6, 251)

for hx in Hx:
        Phase = []
        for j in exchanges:

                #Calculo de la entropia de von Neumann
                valores = [ NumpyvonNeumann( workflow(j1, j2, j3, j, hz, hx, conf) ) for hz in Hz]
                Phase.append(valores)

        #Guardar resultados en un .csv
        df = pd.DataFrame( Phase )
        df.to_csv("Ent"+structure+conf+"jhx"+str(hx)+".csv")
