# SIDM-Sistema-Inteligente-de-Diagnostico-Medico

Aplicación web desarrollada con Django y técnicas de Deep Learning para el análisis, monitoreo y procesamiento inteligente de información mediante modelos de aprendizaje automático.

**Autores:** Alvaro Felipe Avila Vidal, Carlos Eduardo Cabrera Miranda, Juan Jose Urbano Perdomo. 
**Tecnologías principales:** Python, Django, TensorFlow, Transformers  
**Versión de Django:** 4.1

# Instalación y Configuración del Proyecto

## Paso 1: Clonar el repositorio

```bash
git clone https://github.com/AlvaroV19/SIDM-Sistema-Inteligente-de-Diagnostico-Medico.git
```

## Paso 2: Ingresar a la carpeta del proyecto

```bash
cd SIDM-Sistema-Inteligente-de-Diagnostico-Medico
```

## Paso 3: Crear el entorno virtual

### Windows

```bash
python -m venv myenv
```

### Linux / macOS

```bash
python3 -m venv myenv
```

## Paso 4: Activar el entorno virtual

### Windows - PowerShell

```powershell
.\myenv\Scripts\activate
```

### Windows - CMD

```cmd
myenv\Scripts\activate
```

### Linux / macOS

```bash
source myenv/bin/activate
```

## Paso 5: Instalar las dependencias del proyecto

```bash
pip install -r requirements.txt
```

# Librerías utilizadas en SIDM

El proyecto SIDM utiliza las siguientes tecnologías y librerías:

- Django
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Transformers

## Paso 6: Aplicar migraciones

```bash
python manage.py migrate
```

## Paso 7: Ejecutar el servidor de desarrollo

```bash
python manage.py runserver
```

## Paso 8: Abrir la aplicación en el navegador

Abrir en el navegador:

```txt
http://127.0.0.1:8000/
```

# Requisitos del Sistema

- Python 3.11 o superior
- pip
- Git

# Archivo requirements.txt

El proyecto utiliza el siguiente conjunto de dependencias:

```txt
django==4.1
numpy==2.4.4
pandas==3.0.2
matplotlib==3.10.9
seaborn==0.13.2
scikit-learn==1.8.0
tensorflow==2.20.0
tf-keras
transformers==4.57.6
```

# Notas Importantes

- Las carpetas `myenv/` y `.venv/` no se suben al repositorio porque contienen dependencias locales del entorno virtual.
- Todas las dependencias pueden reinstalarse automáticamente usando:

```bash
pip install -r requirements.txt
```

- Si el proyecto utiliza modelos `.h5`, estos pueden gestionarse mediante Git LFS.

# Tecnologías Utilizadas

- Python
- Django
- Deep Learning
- TensorFlow
- Inteligencia Artificial
- Machine Learning
- Transformers/NLP
