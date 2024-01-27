import numpy as np
import numpy.linalg as la
import pandas as pd
import sys

"""
Constantes utiles.
    dtype: tipo de las variables
    gyro: constante giroscopica
    nub: magneton de borh
"""
dtype = 'float128'
gyro = 2.0
nub = 5.7883818066e-2 #meV/T

"""
Funcion para contar cuantos estados con energia val (float) hay en el arreglo l
de energias.
input:
    val: numero real.
    l: lista de valores de valores propios.
output:
    i: numero de valores iguales a val.
"""
def count_rep(val, l):
      aux = np.round(l, 6)
      for i, ee in enumerate(aux):
            if np.abs(ee -val)>1e-6:
                  return i


"""
Funcion para calcular la entropia de von Neumann para un arreglo de valores propios positivos no nulos
input:
    e: vector de valores propios positivos no nulos.
output:
    aux: entropia de von neumann
""" 
def NumpyvonNeumann(ee):
    aux = np.sum( -ee*np.log(ee, dtype=dtype), dtype=dtype )
    return aux


"""
Funcion para calcular la traza parcial de una matriz cuadrada
input:
    rho: matriz cuadrada (matriz densidad)
    size_sub: cantidad de espines del subsistema
    spin_val: Valor del spin (0.5, 1, 1.5 ,...)
output:
    sub_system: matriz resultante de aplicar la traza parcial
""" 
def NumpyPartialTraceLR(rho, size_sub, spin_val):
    spin = int(2*spin_val +1)
    size_partition = np.power(spin, size_sub)
    sub_system = np.zeros((size_partition, size_partition))

    for i in range( int(rho.shape[0]/size_partition) ):
        sub_system = sub_system + rho[size_partition*i: size_partition*(i+1), size_partition*i:size_partition*(i+1)]
    return sub_system


"""
Funcion para calcular los valores y vectores propios
input:
    H: matriz cuadrada hermitica 
output:
    ee: vector con valores propios
    vv: matriz con los vectores propios (para acceder a los vectores usar vv[:,i])
""" 
def Numpyget_eigen(H):
    ee, vv= np.linalg.eigh(H)
    return ee, vv


"""
Funcion para construir el hamiltoniano de la molecula 2-3Ni
input:
    params: lista con los valores de las constantes necesarias para construir el hamiltoniano.
        la lista se compone po J1, J2, J13, J, h_z, h_x, flag, lo importante es la flag que indica 
        que configuracion de acople se usa.
output:
    H: matriz del hamiltoniano
""" 
def hamiltonian(params):
    #J1, J2, J13, J, h, flag
    if params[6] == 1:
        H = -2*params[0]*Int1 -2*params[1]*Int2 -2*params[2]*Int3 -2*params[3]*IntConf1 -gyro*nub*params[4]*OZ -gyro*nub*params[5]*OX
    elif params[6] == 2:
        H = -2*params[0]*Int1 -2*params[1]*Int2 -2*params[2]*Int3 -2*params[3]*IntConf2 -gyro*nub*params[4]*OZ -gyro*nub*params[5]*OX
    elif params[6] == 3:
        H = -2*params[0]*Int1 -2*params[1]*Int2 -2*params[2]*Int3 -2*params[3]*IntConf3 -gyro*nub*params[4]*OZ -gyro*nub*params[5]*OX
    H = np.real(H) 
    return H

"""
Matrices de pauli de spin 1
""" 
sI = np.array( [ [1,0,0], [0,1,0], [0,0,1] ] ,dtype='float64')
sX = (1.0/np.sqrt(2))*np.array( [ [0,1,0], [1,0,1], [0,1,0] ], dtype='float64') 
sY = (1.0/np.sqrt(2))*np.array( [ [0,-1j,0], [1j,0,-1j], [0,1j,0] ], dtype='complex64') 
sZ = np.array( [ [1,0,0], [0,0,0], [0,0,-1] ], dtype='float64') 


"""
Estructuras de los acoples
""" 
Int1 =  np.kron(np.kron(np.kron(sX,sX),sI), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sY,sY),sI), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sZ,sZ),sI), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sX,sX),sI)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sY,sY),sI)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sZ,sZ),sI))

Int2 =  np.kron(np.kron(np.kron(sI,sX),sX), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sI,sY),sY), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sI,sZ),sZ), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sI,sX),sX)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sI,sY),sY)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sI,sZ),sZ))

Int3 =  np.kron(np.kron(np.kron(sX,sI),sX), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sY,sI),sY), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sZ,sI),sZ), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sX,sI),sX)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sY,sI),sY)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sZ,sI),sZ)) 

OZ =    np.kron(np.kron(np.kron(sZ,sI),sI), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sI,sZ),sI), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sI,sI),sZ), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sZ,sI),sI)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sI,sZ),sI)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sI,sI),sZ)) 

OX =    np.kron(np.kron(np.kron(sX,sI),sI), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sI,sX),sI), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sI,sI),sX), np.kron(np.kron(sI,sI),sI)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sX,sI),sI)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sI,sX),sI)) +\
        np.kron(np.kron(np.kron(sI,sI),sI), np.kron(np.kron(sI,sI),sX)) 

IntConf1 =  np.kron(np.kron(np.kron(sI,sI),sX), np.kron(np.kron(sX,sI),sI)) +\
            np.kron(np.kron(np.kron(sI,sI),sY), np.kron(np.kron(sY,sI),sI)) +\
            np.kron(np.kron(np.kron(sI,sI),sZ), np.kron(np.kron(sZ,sI),sI))

IntConf2 =  np.kron(np.kron(np.kron(sI,sI),sX), np.kron(np.kron(sI,sX),sI)) +\
            np.kron(np.kron(np.kron(sI,sI),sY), np.kron(np.kron(sI,sY),sI)) +\
            np.kron(np.kron(np.kron(sI,sI),sZ), np.kron(np.kron(sI,sZ),sI)) 

IntConf3 =  np.kron(np.kron(np.kron(sI,sX),sI), np.kron(np.kron(sI,sX),sI)) +\
            np.kron(np.kron(np.kron(sI,sY),sI), np.kron(np.kron(sI,sY),sI)) +\
            np.kron(np.kron(np.kron(sI,sZ),sI), np.kron(np.kron(sI,sZ),sI)) 

"""
Exchanges moleculares de los dos tipos de moleculas (3D y 1D)
""" 
J3D = [1.49, 1.49, -0.89]
J1D = [-0.08, -0.08, 0.0]
