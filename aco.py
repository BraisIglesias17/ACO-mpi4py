#!/usr/bin/env python
from ctypes.wintypes import PINT
from mpi4py import MPI
import numpy as np


##CONS
alpha=5
beta=1
evaporation_rate=0.5
Q=1
NUM_ITER=1000
INITIAL_PHEROMONE_VALUE=1
Nfil=5
Ncol=5

comm = MPI.COMM_WORLD
numProcs = comm.Get_size()
miRango = comm.Get_rank()

#Validacion de numero de procesos
if numProcs>100:
    if miRango==0:
        print(f'ERROR: Demasiadas hormigas')
    quit()

#Creamos ventana
if miRango==0:
    #Construyo el mapa de pheromonas y el mapa de comida
    mapa = np.full((Nfil, Ncol), INITIAL_PHEROMONE_VALUE,dtype=np.float64)
    comida = np.full((Nfil, Ncol),0,dtype=np.bool_)    

    #Creo las estructuras de datos para almacenar el camino final y la longitud del mismo
    best_path=np.full((Nfil, Ncol),0,dtype=np.int32)
    best_length=np.full(1,0,dtype=np.int32)

    comida[4][4]=True
    comida_size=comida.itemsize
    mapa_size=mapa.itemsize
    best_path_size=best_path.itemsize
    best_length_size=best_length.itemsize
else:
    best_path=None
    comida=None
    comida_size=1
    mapa=None
    mapa_size=1
    best_path_size=1
    best_length=None
    best_length_size=1

#Crea ventana de compartición del mapa de pheromonas
mapa_win=MPI.Win.Create(mapa,mapa_size,comm=comm)
#Crea ventana de compartición de la comida
comida_win=MPI.Win.Create(comida,comida_size,comm=comm)
#Crea ventana de compartición del mejor camino
best_path_win=MPI.Win.Create(best_path,best_path_size,comm=comm)
#Crea ventana de comparticón de la longitud del mejor camino
best_length_win=MPI.Win.Create(best_length,best_length_size,comm=comm)

#Sincronización de procesos antes de comenzar
comm.Barrier()

for i in range(0,NUM_ITER,1):
    #print(f"Soy proceso {miRango} y iteracion {i}")

    #Se crean las estructuras de datos para almacenar los valores de memoria compartida
    pheromones=np.zeros((Nfil, Ncol),dtype=np.float64)
    food=np.zeros((Nfil, Ncol),dtype=np.bool_)
    best_path=np.full((Nfil, Ncol),0,dtype=np.int32)
    best_length=np.full(1,0,dtype=np.int32)

    #Se lee el mapa de pheromonas de memoria compartida
    mapa_win.Lock(0)
    mapa_win.Get([pheromones,MPI.DOUBLE],target_rank=0)
    mapa_win.Unlock(0)
    #Se lee el mapa de pheromonas de comida de memoria compartida
    comida_win.Lock(0)
    comida_win.Get([food,MPI.BOOL],target_rank=0) 
    comida_win.Unlock(0)

    #Se inicializa la matriz de ubicaciones visitadas, y se coloca la posicion inicial (0,0) a True
    visited=np.full((Nfil,Ncol),False,dtype=np.bool_)
    current_point = (0,0) # empieza en el primer punto
    visited[current_point] = True
    
    #Se crea una lista que almacenará los puntos del camino, se inicializa con el punto incial
    path = [current_point]
    path_length = 0

    #Esta variable se utiliza para controlar cuando una hormiga llega a un punto donde no puede moverse y no es una posición de comida, esto se produce ya que las hormigas no pueden pasar dos veces por el mismo
    deadEnd=False

    #Bucle que hace que la hormiga vaya avanzando hasta encontrar comida o llegar a un DEADEND
    while not food[current_point]:
    
        #lista con los puntos sin visitar
        unvisited=[]
    
        #bucle para introducir en la lista los puntos sin visitar, comprobando con sus vecinos
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                x, y = current_point[0] + dx, current_point[1] + dy
                if 0 <= x < visited.shape[0] and 0 <= y < visited.shape[1]:
                    if visited[x][y]==False:
                        unvisited.append((x,y))
        #Si no le quedan más puntos por visitar se encuentra en un deadEnd
        if len(unvisited)==0:
            deadEnd=True
            break
        #Array que almacenará las probabilidades            
        probabilities = np.zeros(len(unvisited))          

        #para cada punto sin visitar se calcula su probabilidad en función del numero de pheromonas que tiene 
        for i, unvisited_point in enumerate(unvisited):
            probabilities[i] = pheromones[unvisited_point]**alpha
        
        #Normalización de las probabilidades
        probabilities /= np.sum(probabilities)
             
        #Se escoge el siguiente punto en funcion de la probabilidad
        next_point = np.random.choice(np.arange(0,len(unvisited),1), p=probabilities)
        next_point=unvisited[next_point]
        
        #Se introduce el siguiente punto al camino y se marca como visitado
        path.append(next_point)
        #Se supone la distancia es igual entre todos los puntos contiguos, se podría hacer una versión con otra matriz de distancias entre puntos (PROBLEMA DEL VIAJANTE)
        path_length += 1
        #Se marca el siguiente punto como visitado y se actualiza
        visited[next_point] = True
        current_point = next_point
        #print(f'\tSiguiente punto: {current_point}')
    
    #Se crea una región crítica para actualizar el mejor camino, esto sólo lo puede estar ejecuanto un proceso en un instante concreto
    best_path_win.Lock(0)   

    best_path_win.Get([best_path,MPI.INT],target_rank=0) 
    best_length_win.Get([best_length,MPI.INT],target_rank=0) 

    if (path_length < best_length[0] or best_length[0] == 0) and not deadEnd:
        #print(f'\tNueva mejor Ruta: {path}')
        best_path=np.full((Nfil, Ncol),0,dtype=np.int32)
        #Se dibuja el camino en memoria compartida
        for point in path:
            best_path[point]=1
        best_path_win.Put([best_path,MPI.INT],target_rank=0)

        #Se actualiza la longitud
        best_length[0]=path_length
        best_length_win.Put([best_length,MPI.INT],target_rank=0)
    best_path_win.Unlock(0)
    
    #Se vuelve a implementar una región crítica para actualizar las pheromonas y que no se pierda ningún cambio
    mapa_win.Lock(0)
        
    mapa_win.Get([pheromones,MPI.DOUBLE],target_rank=0)
    if not deadEnd:    
        for point in path:
            pheromones[point]+=Q/path_length
        mapa_win.Put([pheromones,MPI.DOUBLE],target_rank=0)   
    mapa_win.Unlock(0) 

    #Una vez todos los procesos terminaron la iteracion el proceso 0 lee la matriz y la muestra
    if miRango==0:
        #Degrada las pheromonas
        mapa_win.Lock(0)
        mapa_win.Get([pheromones,MPI.DOUBLE],target_rank=0)
        pheromones*=evaporation_rate
        mapa_win.Put([pheromones,MPI.DOUBLE],target_rank=0)
        #  INSTRUCCION DE PUT
        mapa_win.Unlock(0)

    #Establezco una barrera de sincronizacion de procesos
    #mapa_win.Fence()
    comm.Barrier()

#mapa_win.Fence()
comm.Barrier()    

best_path=np.full((Nfil, Ncol),0,dtype=np.int32)
best_length=np.full(1,0,dtype=np.int32)
best_path_win.Lock(0)
best_path_win.Get([best_path,MPI.INT],target_rank=0)
best_path_win.Unlock(0)


#Se libera la memoria
mapa_win.Free()
comida_win.Free()

if miRango==0:
    print(f"\n ----------------- \n EL MEJOR CAMINO ES :\n {best_path} \n Longitud del mejor camino : {np.count_nonzero(best_path == 1)} \n -----------------------------")