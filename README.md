# Von neumann entropy de dos molecules 3Ni acopladas

Repositorio con los codigos para calcular la entropia de von neumann para dos molecules de 3Ni acopladas.


# Ejecutar codigo
Para ejecutar los codigos, se tiene que estar en el directiorio base del proyecto.

`
python3 <archivo> <tipo de estructura> <configuracion>
`

Los datos calculados se almacenan en una carpeta llamada datos

# Estructuras de los archivos
La parte mas importante es el base.py en donde se definen todas las funciones para construir y calcular elementos imporantes

# Editar archivos
La idea de estos archivos es que sirvan como template para estudiar otras estructuras, como es esperable, una de las primeras cosas que se tienen que cambiar, son las funciones y elementos usandos para la construccion del hamiltoniano, lo siguiente es editar el flujo para que reciba los parametros de la nueva estructura y itere sobre los nuevos rangos y cantidades.