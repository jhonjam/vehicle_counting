#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import math
import tensorflow as tf         #conda install -c anaconda tensorflow==1.15
import cv2
print (cv2.__version__)
print(tf.__version__)


from time import sleep
from scipy.spatial import distance as dist
from scipy.spatial.distance import cdist
from collections import deque
from scipy.spatial import distance

font = cv2.FONT_HERSHEY_SIMPLEX


# ### Parametros de Deep SORT

# In[2]:


# importar todos los archivos de la carpeta de deep sort
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker  # pip install --user scikit-learn==0.22.2
from tools import generate_detections as gdet


# In[3]:


# definicion de parametros 
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 1.0
# inicializacion de parametros de deep sort
#model_filename = 'deep_sort/model_data/market1501.pb' veryjhon.pb
model_filename = 'deep_sort/model_data/veryjhon.pb'
encoder = gdet.create_box_encoder(model_filename,batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)


# # Parametros de YOLO

# In[4]:


net = cv2.dnn.readNet('yolov4_32_e3.weights', 'yolov4_32_e3.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
classes = []
coconames='coco.names'
with open(coconames,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
print(classes)


# In[5]:


# funcion para obtener los nombres de los niveles de la red neutonal
def obtener_nombre_salida(net):
    # Get the names of all the layers in the network
    nombre_niveles = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [nombre_niveles[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# In[6]:


# funcion para definir el color de cada clase, esta funcion trabaja con las classeID que son reconocidas,
#que envian un valor para la respectiva clase, 0,1,2,3 ..n clases. Mirar el archivo coco, que tiene los objetos organizados.
def color_classe(classes):
    if classes == 0: # car 
        color = (100,100,255) 
        return color
    elif classes == 1:  # bus
        color = (255,100,100)
        return color
    elif classes == 2:  # truck
        color = (255,255,100)
        return color
    elif classes == 3:  # van
        color = (240,80,240)
        return color
    elif classes == 4:  # motorbike
        color = (100,255,255)
        return color


# ## Funciones para definir la intercepcion entre las lineas del conteo

# In[7]:


# funciones para determinar y/o encontrar la ecuacion de la recta a partir de dos puntos y verificar si hay intercepcion
#https://stackoverflow.com/questions/29791075/counting-the-point-which-intercept-in-a-line-with-opencv-python

Params = collections.namedtuple('Params', ['a','b','c']) #to store equation of a line


def calcParams(point1, point2): #line's equation Params computation, recibe el centroide pasado y en actual
    if point2[1] - point1[1] == 0:
        a = 0
        b = -1.0
    elif point2[0] - point1[0] == 0:
        a = -1.0
        b = 0
    else:
        a = (point2[1] - point1[1]) / (point2[0] - point1[0]) # esta parece ser la pendiente m=(y2-y1)/(x2-x1)
        b = -1.0

    c = (-a * point1[0]) - b * point1[1] # Ecuacion General ax+by+c=0, si despejo c= -ax-by
                                         # modelo punto pendiente y-y1 = m(x-x1)
    return Params(a,b,c)

def areLinesIntersecting(params1, params2, point1, point2):
    global vab
    det = params1.a * params2.b - params2.a * params1.b
    if det == 0:
        return False #lines are parallel
    else:
        x = round((params2.b * -params1.c - params1.b * -params2.c)/det)
        y = round((params1.a * -params2.c - params2.a * -params1.c)/det)
        if x <= max(point1[0],point2[0]) and x >= min(point1[0],point2[0]) and y <= max(point1[1],point2[1]) and y > min(point1[1],point2[1]):
            #print("intersecting in:", x,y)
            #cv2.circle(frame,(int(x),int(y)),4,(0,0,255), -1) #intersecting point
            return True #lines are intersecting inside the line segment
        
        else:
            return False #lines are intersecting but outside of the line segment

def distancia_euclidiana(p1,p2):
    x = dist.euclidean(p1,p2)
    return x

def area_conteo(punto_objeto,xend,xstar,ystar):
    area_conteo= (xend-xstar)*(punto_objeto[1][1]-ystar)
    if area_conteo<0:
        return True
    else:
        return False

def determinar_conteo_intercepcion_linea(centroides_consecutivos,intercept_line_params):
    # esta funcion recibe los centros actuales del bb, los parametros de abc de la linea de conteo por carril,  
    line_params = calcParams(centroides_consecutivos[0], centroides_consecutivos[1]) # encunetra los parametros abc de la linea que forma los centroides consecutivos
    resultado_interseccion = areLinesIntersecting(intercept_line_params,line_params,centroides_consecutivos[0], centroides_consecutivos[1]) # determina si hubo intercepion entre las dos lineas
    #distancia_Euclidina = distancia_euclidiana(centroides_consecutivos[0], centroides_consecutivos[1]) 
    return resultado_interseccion#,distancia_Euclidina
    
    


# ### Linea de conteo

# In[8]:


# se define la posicion de la linea de conteo dentro del ROI y se encunetra los parametos abc de ese lina para luego
# verificar si en ella se esta intepcentado los centroides consecutivos de los objetos, SOLO SE HACE UNA SOLA VEZ
pos_linha1 = 226
punto1= (141, pos_linha1)
punto2= (1056, pos_linha1)
intercept_line_params1 = calcParams(punto1, punto2) # encontrar la ecuacion de la linea de conteo
# rango para que verifique si existe intercepcion entre las dos lineas
y_arriba = pos_linha1+50
y_abajo = pos_linha1-50


# # Seleccion del Roi

# In[9]:


# crear una macacara para el ROI
cap = cv2.VideoCapture('VideosPruebaAlgoritmo/centro/centro_manana2.mp4')
primer_frame=False # inicializar el primer frame para escoger RoI
cap.set(cv2.CAP_PROP_FRAME_COUNT, 10-1) 
res, frame = cap.read()
imagen = frame
cap.release()
#cv2.imshow('imagen',imagen)
mask = np.zeros(imagen.shape, dtype=np.uint8)
# vertices = np.array([[0,252],[179,90],[570,90],[639,478],[0,478]
#                          ],dtype=np.int32)
# x=cv2.fillPoly(mask,[vertices],(255,255,255)) # x el la masacara en rgb que contiene el tamano original de la imahne de entrada
                                              # solo que muestra ROI y los demas pixeles en negro

vertices = np.array([[0,720],[0,361],[371,71],[939,71],[1279,474],[1279,720]
                         ],dtype=np.int32)
x=cv2.fillPoly(mask,[vertices],(255,255,255))    

# mask = cv2.bitwise_and(imagen, imagen, mask=np.float32(x))
mask = cv2.bitwise_and(imagen,x)
cv2.line(mask, punto1, punto2, (0, 0, 255), 3)


cv2.imshow('img',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Regiones

# In[10]:


#regiones  0 (xu,yu,xd,yd)       1    

#     cv2.rectangle(frame, (0,373), (1280,720), (0, 255, 150), 1)
#     cv2.rectangle(frame, (173,175), (684,315), (0, 255, 150), 1)
#     cv2.rectangle(frame, (691,175), (1100,315), (0, 255, 150), 1)
regiones=[(0,373,1280,720,'R1')]
regionsalida=[(173,175,684,315,'R2'),(691,175,1100,315,'R3')]

# ______________ Tipo de desplazamiento que se pueden presentar en line recta
mo= [('R1','R2','des1_2'),('R1','R3','des1_3')]
#criar variaveis que almacenan o contador de desplazamento por clase
for c in classes:
    for j in mo:
        exec('{}_{}_{}'.format(j[0],j[1],c)+'=0')
        


# # crear .txt com el conteo por regiones

# In[11]:


def createTxt(classes, mo):
    f= open("regiones.txt","w+")
    f.write('      ')

    for c in classes:
        if c =='truck':
            f.write(' tru '.format(c))
        elif c =='motorbike':
            f.write(' mot '.format(c))
        else:
            f.write(' {} '.format(c))
    f.write('\n')

    for j in mo:
        f.write('{}_{}'.format(j[0],j[1])+'   ')
        for c in classes:  
            f.write(str(eval('{}_{}_{}'.format(j[0],j[1],c)))+'    ')
        f.write('\n')

    f.write('\n') 
    f.write('\n')
    f.write('tru = truck '.format(c)+'\n')     
    f.write('mot = motorbike '.format(c)+'\n')        
    f.close()


# ## Funciones para determinar los movimiento a travez de la autopista

# In[12]:


def regionesEntrada(lisIdsIdentificados,colums,lisIdsEliminados,regiones,frame_numero,TT,lisIdsNoIdentificados):
    # este for elimina los ides ya identificados por ids, es decir las columnas del dataframe que estan asociadas a cada vehiculo
    for i in lisIdsIdentificados: # si i=0 esta en esa lista, esa valor sera eliminado
        colums.remove(i)
    
    for i in lisIdsEliminados:
        colums.remove(i)

    for i in colums: #0 1 3  # este contiene el numero de vehiculos detectados por el momento que estan el del dataframe
        centroCuestion = df.iloc[frame_numero][i] # se accesa a los centros de las detecciones actuales del dataframe

        for r in range(len(regiones)):  # se realiza la verificacion si el centro actual esta en algunas de las regiones
            if centroCuestion[:2] != '': # si es valor de la columna es diferente de vacio
#                 print(f'entro com  {i} centroid {centroCuestion[:2]}')
                if zona(centroCuestion[:2],regiones[r][:2],regiones[r][2:]):  #pasa el centro, la esquina superior izquierda de la region, esquina inferior derecha region 
                    TT.append((i,regiones[r][4])) #si el retorno de la funcion zona es true, agrega el id del vehiuclo (columna) junto a la region donde entro
                    lisIdsIdentificados.append(i) #se agrega la lista de ids identificados, que luego se utiliza para no agregarlos de nuevo 
        
        #garantizar que ides que no tuvieron una entrada sean eliminados despues de 5 frames           
        if centroCuestion[:2] == '':
            lisIdsNoIdentificados.append(i)
#             print(f'lisIdsNoIdentificados {lisIdsNoIdentificados}')
            if lisIdsNoIdentificados.count(i)== 5:
                lisIdsEliminados.append(i)
                
    return  lisIdsNoIdentificados,lisIdsIdentificados,TT

def regionesSalida(frame_numero,cdf,regionsalida,TTs,mo,lisIdsIndentSalida): 
    # esta funcion recive el frame actual, el numero de columnas del datafreme, las regiones de salida, 
    # el vector TTs que con la dupla del id vehiculo y la region donde entro, los movimientos posibles (mo), lista de ids identificados de salida   
    columDataframe=list(range(cdf)) # se genera una lista con el numero de columnas del dataframe actual
    
    # este for elimina los ides que ya salieron, es decir las columnas del dataframe que dejaron de llenarse
    # si i=0 esta en esa lista, esa valor sera eliminado
    for j in lisIdsIndentSalida: 
        columDataframe.remove(j)
    
    for i in columDataframe:   
        centroCuestionS = df.iloc[frame_numero][i] # obtiene los centroides actuales
        for r in range(len(regionsalida)):
            if centroCuestionS[:2] != '':
                if zona(centroCuestionS[:2],regionsalida[r][:2],regionsalida[r][2:]): #verifica si estan en las regiones de salida
                    for t in range(len(TTs)): # se va a recorrer la lista TT para verificar si id esta dentro de este  
                        if TTs[t][0]== i:     # si este id esta en la lista TT=[(id,region),...] se va a verificar el movimiento
                            for mos in mo:
                                mov=checkMovement(TTs[t][1], regionsalida[r][4], mos)   #r guarda la posicion por donde salio
                                # mos contiene una lista con tres valores, dos para las zonas y uno para el string del movimiento
                                if mov != None:
                                    #print(f'centros regiones {centroCuestionS[:2]} {regionsalida[r][:2]} {regionsalida[r][2:]}')
                                    lisIdsIndentSalida.append(i)
                                    Movimientos.append((i,mov,centroCuestionS[2]))
    return Movimientos


# In[13]:


# funcion para verificar si un centro esta dentro del las 
# regiones de intercepcion
def zona(centro_actual,p_s_i,p_i_d): # punto superior izquierdo, punto inferior derecho
    return (centro_actual[0]>p_s_i[0] and centro_actual[0]<p_i_d[0] and 
           centro_actual[1]>p_s_i[1] and centro_actual[1]<p_i_d[1])


def verificacion_fuente(v,TT):   # recive el id del vehiculo o columna
    for i in range(len(TT)):
        if TT[i][0] == v:
            return TT[i][1]
        
def checkMovement(rs, rd, mo):
    if mo[0] == rs and mo[1] == rd: # compara si los 
        return mo[2]
    


# ## Funcion para llamar las prediciones por deepsort

# In[14]:


# para cada clase asocarle un color diferente, 
colors = np.random.uniform(0,255,size=(len(classes),3))
def predicion_deepsort(boxes,names,scores,frame):
    # se conbierten las informacion de los objetos e tipo array
    boxes1=boxes
    boxes = np.array(boxes) 
    names = np.array(names)
    scores = np.array(scores)
    features = np.array(encoder(frame, boxes))
    
    # llamamos el metodo de tracking, recibe un numpy array de las detecciones y debuelve los posibles 5 
    # valores que corresponden a bb predicho por kalman y el id asociado
    
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
    #detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                   #zip(converted_boxes, scores[0], names, features)]
    #detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, 
                  #feature in zip(boxes, scores, names, features)]

    tracker.predict()  # call the tracker
    tracker.update(detections)
    
    centroidesSort=[]
    indexIDs = [] # contiene los ids de deepsort detectado en el frame actual    
    j=0
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        
        indexIDs.append(int(track.track_id))
        numero_total_ids.append(int(track.track_id))
        bbox = track.to_tlbr() # se obtiene la posicion del box predicho
        center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
        centroidesSort.append((center))
        
        #class_name = track.get_class()  # estas lineas de codigo son para graficar las prediciones de sort
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.5, (255,255,255),2)        
        #center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
        puntos_centroides_sort[track.track_id].append(center)
        j+=1
        
        #dibujar las lineas de seguimiento
#         for j in range(1, len(puntos_centroides_sort[track.track_id])):
#             if puntos_centroides_sort[track.track_id][j - 1] is None or puntos_centroides_sort[track.track_id][j] is None:
#                 continue
#             thickness = int(np.sqrt(64 / float(j + 1)) * 2)
#             cv2.line(frame,(puntos_centroides_sort[track.track_id][j-1]), (puntos_centroides_sort[track.track_id][j]),(color),thickness)
#             #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)
        


# ## Funcion para la perdida de deteccion

# In[15]:


def perdida_deteccion(listaCentroidesPasados,centroidesYolo,almacenaDeteccionesPerdidas2):
    if len(listaCentroidesPasados)> len(centroidesYolo):
        # esta condicion es necesaria para cuando el vector de yolo es vacio []
        if len(centroidesYolo)==0:
            almacenaDeteccionesPerdidas2=listaCentroidesPasados
        
        else:
            distanciEuclideana_perdidas = np.zeros((len(centroidesYolo), len(listaCentroidesPasados))) # se crea un matriz con el tamano de centros y ids asociados
            dicionario_distancias= {} # en este dicinario se guardan todos las distancias con sus respectivos centros
            for i in range(len(centroidesYolo)):
                for j in range(len(listaCentroidesPasados)):
                    centroides_pasados =np.array(listaCentroidesPasados[j][0:2])
                    centroides_pasados_completos=listaCentroidesPasados[j]
                    centroides_actuales = np.array([centroidesYolo[i][0], centroidesYolo[i][1]])   
                    centroides_actuales_completos=centroidesYolo[i]

                    if not listaCentroidesPasados:
                        continue 

                    else:
                        distancia=distance.euclidean((centroides_pasados[0],centroides_pasados[1]),(centroides_actuales[0],centroides_actuales[1]))
                        distanciEuclideana_perdidas[i,j]=distancia
                        dicionario_distancias[distancia]=[centroides_pasados_completos,centroides_actuales_completos]
                    
            f,c=distanciEuclideana_perdidas.shape
            lista_valores_minimos=[]
            
            if f>=2:
                for j in range(c):
                    minimo_valor=np.argmin(distanciEuclideana_perdidas[:,j]) # aqui se coje la primera columna de toda la matriz para verificar la posicion del menor valor
                    distanciaMinima=distanciEuclideana_perdidas[minimo_valor,j] # de aqui se selecciona la posicion del menor valor 
                    lista_valores_minimos.append((minimo_valor,distanciaMinima))

                claves=[]
                for num,i in enumerate(lista_valores_minimos):
                    for j in lista_valores_minimos[num + 1:]:
                        if i[0]==j[0]:
                            if i[1]>j[1]:
                                claves.append(i[1])
                            elif j[1]>i[1]:
                                claves.append(j[1])

                # ya tengo los vaores de ma minima distancia en un dicionario 
                if len(claves)>1:
                    for z,i in enumerate(claves):
#                         print(f'valor de la clves {i}')
                        for j in claves[z + 1:]:
                            if i !=j:
                                x=dicionario_distancias[i]
                                almacenaDeteccionesPerdidas2.append(x[0])
                elif len(claves)<=1:
                    for i in claves:
                        x=dicionario_distancias[i]
                        almacenaDeteccionesPerdidas2.append(x[0])
                    
            # si la matriz de distancias tiene una sola fila, agregue los valores que no se encuentran con el indice minimo
            elif f==1:
                minimo_valor=np.argmin(distanciEuclideana_perdidas[0])
                for i in range(len(distanciEuclideana_perdidas[0])):
                    if i !=minimo_valor:
                        distanciaMinima=distanciEuclideana_perdidas[0][i] 
                        lista_valores_minimos.append((i,distanciaMinima))
                        x=dicionario_distancias[distanciaMinima]
                        almacenaDeteccionesPerdidas2.append(x[0])
                        
    almacenaDeteccionesPerdidas2=set(almacenaDeteccionesPerdidas2)
    return almacenaDeteccionesPerdidas2


# ## Nueva detecion

# In[16]:


def nueva_deteccion(listaCentroidesPasados,centroidesYolo,almacenaNuevasDetecciones):
    bar2=0
    if len(listaCentroidesPasados)< len(centroidesYolo):
        if len(listaCentroidesPasados)==0:
            almacenaNuevasDetecciones=centroidesYolo

        else:
            for i in range(len(centroidesYolo)):
                for j in range(len(listaCentroidesPasados)):
                    if distance.euclidean(centroidesYolo[i][0:2],listaCentroidesPasados[j][0:2])<10:
                        break
                    else:
                        bar2+=1
                        if bar2==len(listaCentroidesPasados):
                            almacenaNuevasDetecciones.append(centroidesYolo[i])
                            bar2=0
                bar2=0
    return almacenaNuevasDetecciones
  


# # Funcion general para la deteccion y conteo 

# In[27]:


#https://github.com/jorgem0/traffic_counter/blob/master/traffic_counter.py#L178

# deep sort
#https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3/blob/master/object_tracker.py
#https://www.youtube.com/watch?v=sYFhZ4Jc8k4
#https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/deep_sort_yolov3/main.py  barbudo
#https://github.com/theAIGuysCode/yolov3_deepsort/blob/master/object_tracker.py   mono
def etiquetado(frame,salida,car_ids,frame_numero,start_time):
    global numero_car,numero_bus,numero_truck,numero_van,numero_motorbike,idsContados
    global varAgregarFila
    global cxx, cyy,totalcars
    global centroides_pasadosGuard,centroides_actualesGuard,centroides_pasados2,centroides_actuales2
    global centros,puntos_dict,track_id,numero_total_ids,puntos_centroides_sort
    global lisIdsIdentificados,lisIdsNoIdentificados,lisIdsEliminados,TT
    global Movimientos
    global listaDataframeAnterior
    global lenYoloPas,numeroCentroidosPasados, almacenaPerdidaYoloPredicionSort,listaCentroidesPasados,almacenaNuevasDetecciones
    global salidaCar,cumplioloscinco
    global centroidosPasadosParaZerarCol
    global vectordespuesdecinco,salidaszeradas,identificadorcentroides,conteoSalida,vectoranalis,vectormins

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIDs = [] # vector para guardar la id de las clases
    confidences = []
    rectangulos = [] # guaradmos los puntos que crean el rectangulo para poder asociarlo a una clase
    
    for out in salida:
        for detection in out:
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence> 0.5: #valor para evaluar si se estan detectando adecandamente los vehiculos
                #print('class id '+str(classID))
                #if classID == 2 or classID==7 or classID==5 or classID==3:
                cx = int(detection[0]*frameWidth)
                cy = int(detection[1]*frameHeight)
                w = int(detection[2]* frameWidth)
                h = int(detection[3]*frameHeight )
                x = int(cx - w/2)
                y = int(cy - h/2)
                #centros.append((cx,cy))

                classIDs.append(classID)# extraer el nombre de la clase que se detecto
                confidences.append(float(confidence))# guaradr que tan confiavle fue la deteccion
                rectangulos.append([x, y, w, h])

    # esta funcion es para suprimir las detecciones erradas o detecciones repetidas
    indices = cv2.dnn.NMSBoxes (rectangulos,confidences, 0.25, 0.40 )

    # Estos vetores son para el conteo
    cxx = np.zeros(len(rectangulos))  # en cada iteracion se va a crear un array de zeros con el tamano de los rectangulos detectados 
    cyy = np.zeros(len(rectangulos))
    lista_clases_detectadas=[]
    detecciones_completas=[]
    centroidesYolo=[]
    
    boxes, scores, names = [], [], [] # estas lista seran usadas por sort
    
    for i in range(len(rectangulos)):
        if i in indices:
            #i = i[0]
            x,y,w,h= rectangulos[i]
            area= ((x+w)-x)*((y+h)-y)
            if area>500:
                x1 = int(w / 2)
                y1 = int(h / 2)
                cx = x + x1
                cy = y + y1    
                detecciones_completas.append([x,y,w+x,h+y,confidences[i]])#,classIDs[i]]) # le agrego dos veces la confidencia
                # OTRA FORMA PARA GUARDAR LAS DETECCIONES PARA DEEPSORT
                boxes.append([x,y,w+x,h+y])
                scores.append(confidences[i])
                names.append(classIDs[i])           

                label11 = '%.1f' % confidences[i]

                if classes:
                    assert(classIDs[i] < len(classes))
                    label1 = '%s%s' % (classes[classIDs[i]], label11)

                lista_clases_detectadas.append(classes[classIDs[i]]) # se guarda en orden las etiquetas de las clases detectadas

                cv2.rectangle(frame,(x,y),(x+w,y+h),color_classe(classIDs[i]),2)
                fontScale = cv2.FONT_HERSHEY_SIMPLEX
                # etiqueta
                #https://github.com/theAIGuysCode/yolov4-custom-functions/blob/master/core/utils.py
                bbox_thick = int(3 * (1280 + 720) / 600)
                bbox_mess = '%s%s' % (classes[classIDs[i]], label11)
                t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=bbox_thick // 2)[0]  # bbox_thick // 2
                c3 = (x + t_size[0], y - t_size[1] - 3)
                cv2.rectangle(frame, (x,y), (np.float32(c3[0]), np.float32(c3[1])), color_classe(classIDs[i]), -1) #filled
                cv2.putText(frame, bbox_mess, (x, np.float32(y - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0),1 , lineType=cv2.LINE_AA)

                centroidesYolo.append((cx,cy,classes[classIDs[i]]))
                cxx[i] = cx  # se llena en cada posicion el valor del centro en x
                cyy[i] = cy

    end_yolo = time.time()
    fps_yolo = end_yolo - start_time
    fps = 1 / fps_yolo

    tiempo_inicial_alg = time.time()
    #__________________ EN ESTA PARTE SE REALIZA LA GRAFICA DE LAS PREDICIONES DE DEEPSORT___________________
    # se llama la funcion de deepsort, en ocaciones la ejecuto para verificar, NO BORRAR 
    #predicion_deepsort(boxes,names,scores,frame)
    
     #___________________________________________________________________________________#

    

    #____________________ACTUALIZAR Y/O ORGANIZAR LAS DETECCIONES PERDIDAS ___________#
    #____________________________ Agregar las detecciones perdidas ______________________#
    #print(f'lista Centroides Pasados {listaCentroidesPasados}')
    #print(f'lista Centroides actuales yolo {centroidesYolo}')
    
    contMenorK=len(centroidesYolo)
    #print(f'numero de centroides pasados {numeroCentroidosPasados} numero de centroides actuales {contMenorK}')
    
    # ____________________________________PERDIDA DE DETECCION ___________________________________
    # cuando la deteccion actual ha perdido algunos objetos se utliza la infomacion del frame anterior para 
    # veridicar cual fue la deteccion perdida o que salio de la escena y guardarla durante 5 frames
    
    bar3=0
    vectorDistanciasMin=[]
    vectorPosMin=[]
    almacenaDeteccionesPerdidas2=[]
    if salidaCar==5:
        cumplioloscinco+=1
        if cumplioloscinco==2:
            salidaCar=0
            cumplioloscinco=0
            
#####Yessi1             
    if salidaszeradas==1:
        conteoSalida+=1
        
#     print(f' conteoSalida {conteoSalida}')   
#     print(vectordespuesdecinco)
    if conteoSalida==10:  #15 a 1 frame
        vectordespuesdecinco=vectordespuesdecinco.remove(vectordespuesdecinco[0])
        if vectordespuesdecinco == None:
            vectordespuesdecinco=[]
        salidaszeradas=0
        conteoSalida=0
        vectoranalis=[]
        vectormins=[]
#############
        
    # llamanos la funcion que nos devuelve la lista de las deteciones que se han perdido, si las hay
    almacenaDeteccionesPerdidas = perdida_deteccion(listaCentroidesPasados,centroidesYolo,almacenaDeteccionesPerdidas2)
    # realizamos un areglo para que los centroides actuales sean en el sigueinte frame los pasados            
    numeroCentroidosPasados=len(centroidesYolo)
    listaCentroidesPasados=centroidesYolo
    
    #agregamos a cxx y cyy las deteciones perdidas si las hay 
    if len(almacenaDeteccionesPerdidas) != 0:
        cxx = cxx[cxx != 0] # elimina todos los valores iguales a cero
        cyy = cyy[cyy != 0]        
        for ap in almacenaDeteccionesPerdidas:
            newsize=len(cxx)
            cxx=np.resize(cxx, newsize+1)
            cyy=np.resize(cyy, newsize+1)
            cxx[newsize]=ap[0]
            cyy[newsize]=ap[1]
            lista_clases_detectadas.append(ap[2])
            #agregar perdida de deteccion a centroides pasados
            if salidaCar<5:
                listaCentroidesPasados.append(ap)
                salidaCar+=1
            elif salidaCar==5:
                vectordespuesdecinco.append(ap)
                salidaszeradas=1
                conteoSalida=1
                
        almacenaDeteccionesPerdidas=[]
    #________________________________________________________________________________________________________________#
        

    #______________________________________________NUEVA DETECCION_____________________________________________________
    # creo que las nuevas detecciones no tiene mucho problema, esta condicion es para verificar una nueva deteccion al comparar el dataframe anterior con el actual
    # esta funcion compara las nuevas deteciones usando el dataframe, no se usa en el codigo general
    #almacenaNuevasDetecciones=[]
    #nuevas_deteciones = nueva_deteccion(listaCentroidesPasados,centroidesYolo,almacenaNuevasDetecciones)
    #__________________________________________________________________________________________________________________#
    

    # ________________________ AGREGAR LOS CENTROS DETECTODOSA AL DATAFRAME Y REALIZA EL CONTEO_________________#
    cxx = cxx[cxx != 0] # elimina todos los valores iguales a cero
    cyy = cyy[cyy != 0]
    min_indiceMDE = [] #lista vacía para luego verificar en el pd qué indices de los centroides ya estan ya estan agregados
    vectorMinimos=[]
    contadordecolumnasZeradasDF=0
    distanciaMinimaVector=[]
    #print(f"valor de cxx, cyy modificado {(cxx,cyy)}")
    
    # vamos a mirar inicialmente si la lista de ids esta vacia 
    if len(cxx):
        #______________________ CREA LOS ID INICIALES Y REGISTRA LOS PRIMEROS CENTROIDES _______________#
        if not car_ids:  # si la lista ids de los carros esta vacia
            for i in range(len(cxx)):
                car_ids.append(i)
                df[str(car_ids[i])]="" #agrega una columna al marco de datos correspondiente a una car id
                # asigna los valores del centroide al marco actual (fila) y carid (columna) accesa a la posicion  frame actual y adiciona un id car actual al datafremaregistra la etiqueta a la que pertenece
                df.at[int(varAgregarFila), str(car_ids[i])] = [cxx[i],cyy[i],lista_clases_detectadas[i]]                  
            totalcars = len(car_ids)
            
        else:
            distanciEuclideana = np.zeros((len(cxx), len(car_ids))) # se crea un matriz con el tamano de centros y ids asociados
            idsTT=[]  # lista para agregar los ids
            for i in range(len(cxx)): # vamos a recorre los nuevos centros para comparar con los ids anteriores asigandos                      
                    for j in range(len(car_ids)):# y los centroides anteriores
                        centroides_pasados = df.iloc[int(varAgregarFila)-1][str(car_ids[j])] # se accesa a los valores cx,xy, clase regsitrados en el frama anterio        
                        centroides_actuales = np.array([cxx[i], cyy[i]]) # se guardan en la variable los valores de cx,cy del frame actual                      
               
                        if not centroides_pasados:# or centroides_pasados!='': # se verifica si el nuevo centroide esta vacio o en caso que aparezca#un nuevo vehiculo (nuevo car_id) continue sin problemas 
                            continue 
                        else:
                            #se llena la matriz con los valores de la distacia de todos los centros, encuentra la distancia euclidianentre todos los valores de la matriz
                            distanciEuclideana[i,j]=distance.euclidean((centroides_pasados[0],centroides_pasados[1]),(centroides_actuales[0],centroides_actuales[1]))

            contadordecolumnasZeradasDF=len(car_ids)-centroidosPasadosParaZerarCol
            distanciaMinimaVector=[]
            vectorMinimos=[]
            vectorColumnasMinimas=[]
        

#####Yessi2#################                
            if len(vectordespuesdecinco)>0:
                distanciaVectorsalvo = np.zeros((len(vectordespuesdecinco),len(cxx)))
                for jv in range(len(vectordespuesdecinco)):
                    s=vectordespuesdecinco[jv]
                    if s not in vectoranalis:
                        for iv in range(len(cxx)): 
                            centroides_actuales = np.array([cxx[iv], cyy[iv]]) 
                            distanciaVectorsalvo[jv,iv]=distance.euclidean((s[0],s[1]),(centroides_actuales[0],centroides_actuales[1]))
                            vectoranalis.append(s)
#                         print(f' distanciaVectorsalvo  {distanciaVectorsalvo}')
                        minu=np.argmin(distanciaVectorsalvo[jv])
#                         print(minu)
                        vectormins.append(minu+contadordecolumnasZeradasDF)
############################            

            dicionario_deteciones_cercanas_anterior={}
            for j in range(len(car_ids)):             
                if np.all(distanciEuclideana[:, j] == 0):
                    vectorColumnasMinimas.append(0)  # esta lista se llena para luego recorer en las mismas posiciones del dataframe
                    continue
                else:
                    dicionario_deteciones_cercanas_actual={}
                    posicion_minima_distancia=np.argmin(distanciEuclideana[:,j]) # aqui se coje la primera columna de toda la matriz para verificar la posicion del menor valor
                    valor_distancia_minima=distanciEuclideana[posicion_minima_distancia,j] # de aqui se selecciona la posicion del menor valor
                    
                    vectorColumnasMinimas.append(posicion_minima_distancia)
                    
                    dicionario_deteciones_cercanas_actual[valor_distancia_minima]=[posicion_minima_distancia,j] # agrego en cada interacion al dicionario la menor distancia y la posicion en la matriz donde se encuentra
                    #dicionario_detecciones_totales[valor_distancia_minima]=posicion_minima_distancia

                    if len(dicionario_deteciones_cercanas_anterior)>=1:                
                        for s,v in dicionario_deteciones_cercanas_actual.items():
                            for n, k in dicionario_deteciones_cercanas_anterior.items(): # esta funcion devuelve la clave y el valor de un dicionario
                                # verificacion si el valor encontrado contiene el mismo valor de la de la fila
                                #print (f'valor que cotine los dicionarios de distancias minimas de v {v} y k {k}')
                                if v[0]==k[0]:
                                    if s>n:
                                        distanciEuclideana[v[0],v[1]]=100
                                    elif n>s:
                                        distanciEuclideana[k[0],k[1]]=100             

                    dicionario_deteciones_cercanas_anterior[valor_distancia_minima]=[posicion_minima_distancia, j]#dicionario_deteciones_cercanas_actual

            #print(f'nueva matriz de distancias euclidinas {distanciEuclideana}')
            #print(f'vectorColumnasMinimas {len(vectorColumnasMinimas)} {vectorColumnasMinimas}')
            
            for j in range(len(car_ids)):
                minimo=vectorColumnasMinimas[j]
                distanciaMinima=distanciEuclideana[minimo,j] #distanciaMinimaVector[j] # de aqui se selecciona la posicion del menor valor 

                # esta condicion fue realizada cuando los vehiculos estan parados y los centroides son los mismo en N frames 
                centroides_pasados_iguales = df.iloc[int(varAgregarFila)-1][str(car_ids[j])] # se accesa a los valores cx,xy, clase regsitrados en el frama anterior
                if distanciaMinima == 0 and np.all(distanciEuclideana[:, j] == 0):
                    if not centroides_pasados_iguales:
                        continue
                    else:
                        # se compara elemento a elemento de los centros pasados y actuales, si son los mismos cxx pasado y cxx actual agrege ese valor al vector del menor indice 
                        if centroides_pasados_iguales[0]==cxx[minimo] and centroides_pasados_iguales[1]==cyy[minimo]:
                            df.at[int(varAgregarFila), str(car_ids[j])] = [cxx[minimo], cyy[minimo],lista_clases_detectadas[minimo]] 
                            min_indiceMDE.append(minimo)
                        else:
                            continue # continue to next carid

                else:
                    # si la distancia es menor a cierto umbral, se agrega el centro a la misma columna donde fue detectado por primera
                    if distanciaMinima < 90:                                                               #45:
                        df.at[int(varAgregarFila), str(car_ids[j])] = [cxx[minimo], cyy[minimo],lista_clases_detectadas[minimo]] 

                        min_indiceMDE.append(minimo)  # minx_index2  agrega la posicion que encontro con menor distancia euclidiana                    

                        #___________________VERIFICACION SI EXISTE INTERCEPCION ENTRE LOS CENTROS ACTUALES Y PASADOS 
                        centroides_pasados2 = df.iloc[int(varAgregarFila)-1][str(car_ids[j])]
                        centroides_actuales2= df.iloc[int(varAgregarFila)][str(car_ids[j])]

                        # _____________________ FUNCIONES PARA EL CONTEO __________________________________________#
                        # si los valores en Y de los centros pasados y actuales se encuentran 30 pixeles encima y 30 pixeles abajo
                        # se empieza a llamar las funciones para determinar si existe alguna intercepcion entre las lineas generadas
                        # esto se hace con el objetivo de disminuir el tiempo de procesaimeto
                        
                        if  centroides_pasados2!='' and centroides_actuales2!='': 
                            if centroides_pasados2[0:2][1] <= y_arriba and centroides_pasados2[0:2][1] >=y_abajo and centroides_actuales2[0:2][1] <= y_arriba and centroides_actuales2[0:2][1] >=y_abajo: 
                                resultado_interseccion=determinar_conteo_intercepcion_linea((centroides_pasados2[0:2],centroides_actuales2[0:2]),intercept_line_params1) # 
                                etiqueta= centroides_actuales2[2]
                                if j+1 not in idsContados and resultado_interseccion is True and centroides_pasadosGuard!=centroides_pasados2 and centroides_actuales !=centroides_actuales2:
                                    idsContados.append(j+1)
                                    if etiqueta == 'car':
                                        numero_car+=1
                                    elif etiqueta == 'bus':
                                        numero_bus+=1
                                    elif etiqueta =='truck':
                                        numero_truck+=1
                                    elif etiqueta == 'van':
                                        numero_van+=1
                                    elif etiqueta =='motorbike':
                                        numero_motorbike+=1
#####Yessi3                                                   
            if len(vectordespuesdecinco)>0:
                for jh in range(len(vectordespuesdecinco)):
                    pv=vectordespuesdecinco[jh]
                    for ih in range(len(cxx)):
                        if ih not in min_indiceMDE:
                            pc=cxx[ih], cyy[ih]
                            #print(pc,pv)
                            distancever=distancia_euclidiana(pc,pv[:2])      
                            #print('------------------------------')
#                             print(distancever)
                            if distancever<20:
                                print(vectormins)
                                #var= vectormins[jh]-contadordecolumnasZeradasDF-1
                                min_indiceMDE.append(ih)
                                #print(ih)    
                                #print(jh)
                                #print(contadordecolumnasZeradasDF-1)
                                #print(lista_clases_detectadas)
                                df.at[int(varAgregarFila), str(vectormins[jh])] = [cxx[ih], cyy[ih],lista_clases_detectadas[ih]] 
#################            
            # por ejemplo min_indiceMDE=[0,1,2], este vector contiene la posicion de los centroides
            # que presentaron la menor distancia euclidiana, si i no esta en esta lista, se crea una nueva columna
            
            #_______ Cuando los centroides estan en una distancia mayor a 90 se crea una nueva columna __________#

            for i in range(len(cxx)): # vamos a recorrer el tamanho de los cetroides detectodos en el frame actual
                # si el indice no esta en la lista de indices minimos se debe agragar otro id car
                if i not in min_indiceMDE:    
                    df[str(totalcars)] = ""  # create another column with total cars
                    totalcars = totalcars + 1  # adds another total car the count
                    t = totalcars - 1  # t is a placeholder to total cars
                    car_ids.append(t)  # append to list of car ids
                    df.at[int(varAgregarFila), str(t)] = [cxx[i], cyy[i],lista_clases_detectadas[i]]  # add centroid to the new car id

                elif centroides_actuales[0] and not centroides_pasados and not min_indiceMDE:
                    df[str(totalcars)] = ""  # create another column with total cars
                    totalcars = totalcars + 1  # adds another total car the count
                    t = totalcars - 1  # t is a placeholder to total cars
                    car_ids.append(t)  # append to list of car ids
                    df.at[int(varAgregarFila), str(t)] = [cxx[i], cyy[i],lista_clases_detectadas[i]]  # add centroid to the new car id 
    #___________________________________________________________________________________________________________________________#
    '''
    currentcars = 0  # current cars on screen
    currentcarsindex = []  # current cars on screen carid index

    for i in range(len(car_ids)):  # loops through all carids

        if df.at[int(frame_numero), str(car_ids[i])] != '':
            # checks the current frame to see which car ids are active
            # by checking in centroid exists on current frame for certain car id
            currentcars = currentcars + 1  # adds another to current cars on screen
            currentcarsindex.append(i)  # adds car ids to current cars on screen
    
    for i in range(currentcars): 
        curcent = df.iloc[int(frame_numero)][str(car_ids[currentcarsindex[i]])]        
        if curcent:  
            cv2.putText(frame, "ID:" + str(car_ids[currentcarsindex[i]]+1), (int(curcent[0]), int(curcent[1] - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 2)

    '''
    #___________________________VERIFICACION DONDE LOS OBJETOS SE ESTAN MOVIENDO ____________________________#
    # para establecer por donde los vehiculos se estan miviendo A HASTA EL PUNTO B, se toma la informacion del
    # dataframe construido, ya que este contiene todos las posiciones de los vehiculos desde que aparecen hasta que salen de la escena

    fdf,cdf=df.shape # encontramos el numero de filas y columnas del df  
    colums=list(range(cdf)) # se crea una lista con el numero de columnas del dataframe
#     print(lisIdsIdentificados,colums,lisIdsEliminados,regiones,frame_numero,TT)
#     print(lisIdsNoIdentificados)
    lisIdsNI,lisIdsI,TTs=regionesEntrada(lisIdsIdentificados,colums,lisIdsEliminados,regiones,frame_numero,TT,lisIdsNoIdentificados)
#     print(f'lista de ids {lisIdsI} TTS {TTs}')
    
    movimientos= regionesSalida(frame_numero,cdf,regionsalida,TTs,mo,lisIdsIndentSalida)
    #print(f'movimientos {movimientos}')
    
    Movimientos=[]    

    
    #___________________________________________________________________________________#
    
    #print('++++++++++++++++++++++++++')
    centroidosPasadosParaZerarCol=len(centroidesYolo)
    frame_numero += 1
        
    cv2.line(frame, punto1, punto2, (255, 0, 0), 3)
    cv2.putText(frame, "FRAME   : " + str(frame_numero), (12, 40), font, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "FPS Yolo: " + str(round(fps, 2)), (12, 80), font, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Car     : " + str(numero_car), (12, 120), font, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Bus     : " + str(numero_bus), (12, 160), font, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Truck   : " + str(numero_truck), (12, 200), font, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Van     : " + str(numero_van), (12, 240), font, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Motorbike: " + str(numero_motorbike), (12, 280), font, 0.5, (0, 255, 0), 1)

    cv2.rectangle(frame, (0, 373), (1280, 720), (0, 255, 150), 1)
    cv2.rectangle(frame, (173, 175), (717, 312), (0, 255, 150), 1)
    cv2.rectangle(frame, (745, 175), (1100, 315), (0, 255, 150), 1)

    tiempo_final_alg=time.time()
    tiempo = tiempo_final_alg - tiempo_inicial_alg
    
    return frame_numero,frame,movimientos,tiempo


# In[28]:


# llamar el archivo del sort, este archivo no tiene implemtacion con deep
#from sort import *
#mot_tracker=funcaoseguimento()

#______________ Tamanho do video de entrada______________
cap = cv2.VideoCapture('VideosPruebaAlgoritmo/centro/centro_manana2.mp4')
frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#______________ Configuracao para guardar o video de saida com as deteciones _________
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('VideosPruebaAlgoritmo/centro/detecciones_centro_manana.mp4',fourcc, 5.0, (1280,720))

#______________ Crea um pandas data frame con el numero de filas con el numero de frames del video ______
df = pd.DataFrame(index=range(int(frames_count))) #(index=range(int(frames_count)))
df.index.name = "Frames"
varAgregarFila=0 # inicializacion para verificar la posicion para agregar las informacion en cada frame, este valor tiene
                 # el mismo valor de frame_numero
totalcars=0 # este es utilizado para crear el numero de columnas con base en el tamano de Carids del dataframe
car_ids = [] # contiene los valores de las columnas del dataframe

#_______________ Variables para el conteo de vehiculos______
numero_car=0
numero_bus=0
numero_truck=0
numero_van=0
numero_motorbike=0
idsContados=[]


# ______________ Variables para graficar la lista de centros de deepsort ______
puntos_centroides_sort = [deque(maxlen=20) for _ in range(9999)]
track_id = []  # contiene los ids identificados por deepsort
numero_total_ids =[] # contiene el total de ids identificados por deepsort

# ______________ Variables para realizar la comparacion de los centroiedes pasados y actules utilizados en el conteo ____
centroides_pasadosGuard=[]
centroides_actualesGuard=[]
centroides_pasados2=[]
centroides_actuales2=[]

#_______________  Variables para identificar los vehiculos por que zona se estan moviendo ______
lisIdsIndentSalida=[]
Movimientos=[]
lisIdsIdentificados=[]
lisIdsNoIdentificados=[]
lisIdsEliminados=[]
TT=[]



listaDataframeAnterior=[] #guardar los centroides pasados de dataframe 

# NO SE PARA QUE SON
#centros =[]
#IDs_obj_general=[]
#centros_general=[]
#puntosDict = {}
#puntos_dict ={}


frame_numero=0

#perdidas de yolo y sort no ayuda
lenYoloPas=0
numeroCentroidosPasados=0
centroidosPasadosParaZerarCol=0

# perdidas quado yolo no detecta y sort predice una nueva posicion
almacenaPerdidaYoloPredicionSort=[]
listaCentroidesPasados=[]
almacenaNuevasDetecciones=[]

#bandera para determinr la repeticion de salida de los carros
salidaCar=0
cumplioloscinco=0

#perdidas despues de irse
vectordespuesdecinco=[]
salidaszeradas=0
identificadorcentroides=0
conteoSalida=0
vectoranalis=[]
vectormins=[]

#variable para cada 5 frame
cumpliocinco=0

while True:    
    ret, frame = cap.read()
    #print(cumpliocinco)
    if cumpliocinco==3:
        start_time = time.time()
        if frame is None or cv2.waitKey(1) == 27 or frame_numero ==frames_count:
            createTxt(classes, mo)
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #print('frame '+str(frame_numero))
            
        ROI = cv2.bitwise_and(frame,x)
        blob = cv2.dnn.blobFromImage(ROI, 1/255, (512, 512), [0,0,0], 1, crop = False)
        net.setInput(blob)
        salida = net.forward(obtener_nombre_salida(net))

        # Mirar cuanto frames por segundo son ejecutados
        #elapsed_time = time.time() - starting_time
        #fps = frame_numero / elapsed_time  # COMO LO TENIA ANTERIORMENTE


        # Funcion que me devuelve la predicion de la red, ademas de ello le evio la fps para que lo muestre en pantalla
        frame_numero,frame2,movs,tiempo_alg=etiquetado(ROI,salida,car_ids,frame_numero,start_time) #,track_id,puntosDict)
        
        for i in range(len(movs)):
            for j in mo:
                if movs[i][1]==j[2]:
                    exec('{}_{}_{}'.format(j[0],j[1],movs[i][2])+'+=1')        
        
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        if tiempo_alg !=0:
            FPS = 1 / (tiempo_alg)
            #fps = frame_numero / seconds
            cv2.putText(frame2, "FPS ALG   : " + str(round(FPS, 2)), (200, 40), font, 0.5, (0, 255, 0), 1)
        end_general = time.time()
        tiempo_general =end_general - start_time

        FPS_G = 1 / (tiempo_general)
        cv2.putText(frame2, "FPS GENERAL   : " + str(round(FPS_G, 2)), (400, 40), font, 0.5, (0, 255, 0), 1)
        cv2.imshow('frame',frame2)
        video_writer.write(frame2)
        
        varAgregarFila = 1+varAgregarFila  # nuevo

        
                
        cumpliocinco=0
    else:
        cumpliocinco+=1

cap.release()
video_writer.release()
cv2.destroyAllWindows()


# In[ ]:


#df.to_csv('traffic.csv', sep=',')
df.to_excel("output.xlsx") 




