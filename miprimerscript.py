import pandas as pd #Cargamos la librería de pandas
import numpy as np
import seaborn as sns #para graficar
#Obtenemos nuestro dataset
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
print("Veamos la información sobre nuestro dataset de nombre iris")
print(iris.info(), "\n")
print("Las primeras filas son:\n",iris.head(), "\n")
print("Los principales datos estadísticos de las variables númericas son:\n", iris.select_dtypes(np.number).describe(),"\n")
iris["species"] = iris["species"].astype('category') #Cambiamos el tipo de variable que es species
print("Los principales datos estadísticos de la clase a predecir es son:\n", iris.select_dtypes("category").describe())
sns.set_theme(style="ticks") #Para que se vea re bien bonito
print(sns.pairplot(data=iris, hue="species"))