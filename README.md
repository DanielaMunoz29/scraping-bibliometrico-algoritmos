# 📚 Sistema de Análisis Académico - Biblioteca Uniquindío

Sistema integral de análisis bibliométrico y académico con interfaz gráfica moderna. Permite realizar scraping, limpieza, análisis y visualización de artículos académicos de múltiples bases de datos.

## 🎯 Características Principales

- **Scraping Automatizado**: Extracción de artículos de EBSCO e IEEE
- **Autenticación Persistente**: Sistema de cookies con validación automática
- **Limpieza y Unificación**: Procesamiento y eliminación de duplicados
- **Análisis Avanzado**: Múltiples algoritmos de ordenamiento y análisis bibliométrico
- **Visualizaciones**: Gráficos, nubes de palabras, mapas de citaciones y más
- **Interfaz Moderna**: GUI basada en pywebview con diseño responsive

## 📋 Requisitos

- Python 3.12+
- Google Chrome (para scraping con Selenium)
- Credenciales institucionales para EBSCO e IEEE

## 🚀 Instalación

1. Clonar el repositorio y navegar al directorio:
```bash
cd biblioteca_uniquindio
```

2. Crear y activar entorno virtual:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 💻 Uso

### Iniciar la aplicación

```bash
python main.py
```

### Flujo de trabajo

1. **Autenticación**: Login en las bases de datos seleccionadas
2. **Consulta de Disponibilidad**: Verificar cantidad de artículos disponibles
3. **Scraping**: Extraer artículos según criterios de búsqueda
4. **Limpieza**: Unificar y eliminar duplicados
5. **Análisis**: Ejecutar algoritmos de ordenamiento y análisis bibliométrico
6. **Visualización**: Generar gráficos y reportes

## 📁 Estructura del Proyecto

```
biblioteca_uniquindio/
├── main.py                      # Punto de entrada
├── academic_analysis_gui.py     # Backend de la GUI
├── interface.html               # Frontend de la GUI
├── overrides.css               # Estilos personalizados
├── src/
│   ├── scraper/                # Módulos de scraping
│   │   ├── EBSCO.py
│   │   └── IEEEScraper.py
│   ├── data/                   # Procesamiento de datos
│   │   ├── MultiDatabaseCleaner.py
│   │   ├── cookies/           # Cookies de sesión
│   │   ├── csv/               # Archivos CSV generados
│   │   └── unified/           # Datos unificados
│   └── algoritmo/             # Algoritmos de análisis
│       ├── AcademicSortingAnalyzer.py
│       ├── BibliometricVisualizer.py
│       ├── CitationNetworkAnalyzer.py
│       ├── ConceptsCategoryAnalyzer.py
│       ├── CooccurrenceNetworkAnalyzer.py
│       ├── HierarchicalClusteringAnalyzer.py
│       ├── SimilitudTextualClasico.py
│       └── SimilitudTextualIA.py
└── browser_profile/           # Perfil de navegador para scraping
```

## 🔧 Módulos Principales

### Scrapers
- **EBSCO**: Extracción de artículos de bases de datos EBSCO
- **IEEE**: Extracción de artículos de IEEE Xplore

### Algoritmos de Análisis
- **AcademicSortingAnalyzer**: Comparación de algoritmos de ordenamiento
- **SimilitudTextualClasico**: Análisis de similitud con TF-IDF
- **SimilitudTextualIA**: Análisis de similitud con embeddings
- **CitationNetworkAnalyzer**: Análisis de redes de citaciones
- **ConceptsCategoryAnalyzer**: Análisis de conceptos y categorías
- **HierarchicalClusteringAnalyzer**: Clustering jerárquico
- **CooccurrenceNetworkAnalyzer**: Análisis de co-ocurrencias
- **BibliometricVisualizer**: Visualizaciones bibliométricas

### Procesamiento de Datos
- **MultiDatabaseCleaner**: Limpieza y unificación de múltiples fuentes

## 📊 Funcionalidades de Análisis

- Comparación de tiempos de ejecución de algoritmos de ordenamiento
- Análisis de autores más citados
- Redes de citaciones interactivas
- Clustering de artículos por similitud
- Análisis de co-ocurrencia de términos
- Visualizaciones bibliométricas avanzadas
- Generación de reportes en PDF

## 🔐 Autenticación

El sistema utiliza cookies persistentes para mantener sesiones activas:
- Las cookies se almacenan en `src/data/cookies/`
- Validación automática antes de cada operación
- Re-autenticación automática si las cookies expiran
- Soporte para autenticación 2FA (timeout de 60 segundos)

## 📈 Salidas Generadas

- **CSV**: Artículos extraídos y procesados
- **PNG**: Gráficos de comparación y visualizaciones
- **HTML**: Mapas interactivos y redes de citaciones
- **PDF**: Reportes bibliométricos completos

## ⚙️ Configuración

### Mostrar/Ocultar Navegador
La aplicación permite ejecutar el scraping en modo headless o visible según preferencia del usuario.

### Cantidad de Artículos
- Descargar todos los resultados disponibles
- Especificar cantidad personalizada

### Bases de Datos
Selección múltiple de bases de datos para scraping simultáneo.

## 🐛 Solución de Problemas

### Error de autenticación
- Verificar credenciales institucionales
- Asegurar acceso a las bases de datos desde la red institucional
- Completar 2FA dentro del tiempo límite (60 segundos)

### Error de scraping
- Verificar conexión a internet
- Comprobar que Chrome está instalado
- Revisar que las cookies son válidas

### Error de análisis
- Verificar que los archivos CSV tienen el formato correcto
- Asegurar que hay suficientes datos para el análisis

## 👥 Autores

Proyecto desarrollado para el análisis académico de la Biblioteca Universidad del Quindío.

## 📝 Licencia

Este proyecto es de uso académico e institucional.

## 🔄 Actualizaciones

Para mantener el sistema actualizado:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

**Nota**: Este sistema requiere credenciales institucionales válidas para acceder a las bases de datos académicas.
