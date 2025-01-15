from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el dataset con las propiedades asignadas a los estudiantes, incluyendo 'city'
df = pd.read_csv(
    'propiedades_asignadas_con_estudiantes_por_ciudad.csv', low_memory=False)
# Dataset principal con todas las propiedades
df_busqueda = pd.read_csv(
    'propiedades_asignadas_con_estudiantes.csv', low_memory=False)

# Seleccionar las columnas que se usarán para la predicción, incluyendo 'city'
features = ['bedrooms', 'beds', 'Waterfront', 'Elevator', 'Pets allowed', 'Smoking allowed',
            'Wheelchair accessible', 'Pool', 'TV', 'Microwave', 'Internet', 'Heating']

# Incluir la columna 'city' en el DataFrame
df = df[['student_id', 'id', 'city'] + features]

# Convertir la columna 'city' a variables dummies (one-hot encoding)
df = pd.get_dummies(df, columns=['city'], drop_first=True)

# Crear la lista completa de características incluyendo las columnas dummies de 'city'
features += [col for col in df.columns if col.startswith('city_')]

# Agrupar por student_id y calcular las características promedio de las propiedades de cada estudiante
student_features = df.groupby('student_id').mean().reset_index()

# Dividir los datos en características y el target
X = student_features[features]
y = X  # En este caso, predeciremos todas las características

# Entrenar el modelo de regresión lineal múltiple
model = LinearRegression()
model.fit(X, y)

# Página principal de la aplicación


@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sistema de Propiedades para Estudiantes</title>
        </head>
        <body>
            <h1>Sistema de Propiedades para Estudiantes</h1>

            <h2>Recomendación de Propiedad</h2>
            <label for="student_id_recom">Ingrese el ID del estudiante:</label>
            <input type="text" id="student_id_recom" placeholder="Ejemplo: 12345">
            <button onclick="recomendarPropiedad()">Obtener Recomendación</button>
            <h3 id="resultado_recom"></h3>

            <h2>Búsqueda de Propiedades Asignadas</h2>
            <label for="student_id_buscar">Ingrese el ID del estudiante:</label>
            <input type="text" id="student_id_buscar" placeholder="Ejemplo: 98">
            <button onclick="buscarPropiedades()">Buscar Propiedades</button>
            <h3 id="resultado_busqueda"></h3>

            <h2>Búsqueda de Propiedad por ID</h2>
            <label for="property_id">Ingrese el ID de la propiedad:</label>
            <input type="text" id="property_id" placeholder="Ejemplo: 101">
            <button onclick="buscarPropiedadPorID()">Buscar Propiedad</button>
            <h3 id="resultado_propiedad"></h3>

            <script>
                function recomendarPropiedad() {
                    const studentId = document.getElementById('student_id_recom').value.trim();
                    if (studentId) {
                        fetch(`/recomendar_propiedad?student_id=${studentId}`)
                            .then(response => response.json())
                            .then(data => {
                                if (data.error) {
                                    document.getElementById('resultado_recom').innerText = data.error;
                                } else {
                                    document.getElementById('resultado_recom').innerText = 
                                        `La propiedad recomendada para el student_id ${data.student_id} es el id: ${data.recommended_property_id} en la ciudad: ${data.city}`;
                                }
                            })
                            .catch(error => {
                                document.getElementById('resultado_recom').innerText = 'Error al obtener la recomendación';
                            });
                    } else {
                        document.getElementById('resultado_recom').innerText = 'Por favor ingrese un ID de estudiante válido';
                    }
                }

                function buscarPropiedades() {
                    const studentId = document.getElementById('student_id_buscar').value.trim();
                    if (studentId) {
                        fetch(`/buscar_propiedades?student_id=${studentId}`)
                            .then(response => response.json())
                            .then(data => {
                                if (data.error) {
                                    document.getElementById('resultado_busqueda').innerText = data.error;
                                } else {
                                    let resultadoHTML = `<h3>Propiedades asignadas al estudiante ${data.student_id}:</h3>`;
                                    resultadoHTML += "<table border='1'><tr><th>Tipo de Propiedad</th><th>Ciudad</th><th>Código Postal</th><th>Dormitorios</th><th>Camas</th></tr>";
                                    data.propiedades.forEach(propiedad => {
                                        resultadoHTML += `<tr>
                                            <td>${propiedad.property_type}</td>
                                            <td>${propiedad.city}</td>
                                            <td>${propiedad.zipcode}</td>
                                            <td>${propiedad.bedrooms}</td>
                                            <td>${propiedad.beds}</td>
                                        </tr>`;
                                    });
                                    resultadoHTML += "</table>";
                                    document.getElementById('resultado_busqueda').innerHTML = resultadoHTML;
                                }
                            })
                            .catch(error => {
                                document.getElementById('resultado_busqueda').innerText = 'Error al buscar las propiedades';
                            });
                    } else {
                        document.getElementById('resultado_busqueda').innerText = 'Por favor ingrese un ID de estudiante válido';
                    }
                }

                function buscarPropiedadPorID() {
                    const propertyId = document.getElementById('property_id').value.trim();
                    if (propertyId) {
                        fetch(`/buscar_propiedad?id=${propertyId}`)
                            .then(response => response.json())
                            .then(data => {
                                if (data.error) {
                                    document.getElementById('resultado_propiedad').innerText = data.error;
                                } else {
                                    let resultadoHTML = `<h3>Detalles de la propiedad con ID ${data.id}:</h3>`;
                                    resultadoHTML += "<ul>";
                                    for (const [key, value] of Object.entries(data.detalles)) {
                                        resultadoHTML += `<li><strong>${key}:</strong> ${value}</li>`;
                                    }
                                    resultadoHTML += "</ul>";
                                    document.getElementById('resultado_propiedad').innerHTML = resultadoHTML;
                                }
                            })
                            .catch(error => {
                                document.getElementById('resultado_propiedad').innerText = 'Error al buscar la propiedad';
                            });
                    } else {
                        document.getElementById('resultado_propiedad').innerText = 'Por favor ingrese un ID de propiedad válido';
                    }
                }
            </script>
        </body>
        </html>
    ''')

# Ruta para recomendar propiedad


@app.route('/recomendar_propiedad', methods=['GET'])
def recomendar_propiedad():
    student_id = request.args.get('student_id', type=int)
    if student_id not in student_features['student_id'].values:
        return jsonify({'error': f"El student_id {student_id} no existe en el conjunto de datos."}), 404

    student_data = student_features[student_features['student_id']
                                    == student_id][features]
    predicted_features = model.predict(student_data)
    df['distance'] = np.sqrt(
        ((df[features] - predicted_features[0]) ** 2).sum(axis=1))
    recommended_property = df.loc[df['distance'].idxmin()]

    city_columns = [col for col in df.columns if col.startswith('city_')]
    recommended_city = [col.replace(
        'city_', '') for col in city_columns if recommended_property[col] == 1]

    return jsonify({
        'student_id': student_id,
        'recommended_property_id': int(recommended_property['id']),
        'city': recommended_city[0] if recommended_city else 'Desconocida'
    })

# Ruta para buscar propiedades por student_id


@app.route('/buscar_propiedades', methods=['GET'])
def buscar_propiedades():
    student_id = request.args.get('student_id', type=int)
    propiedades = df_busqueda[df_busqueda['student_id'] == student_id]
    if propiedades.empty:
        return jsonify({'error': f"No se encontraron propiedades para el estudiante con ID: {student_id}"}), 404

    columnas_deseadas = ['property_type',
                         'city', 'zipcode', 'bedrooms', 'beds']
    columnas_existentes = [
        col for col in columnas_deseadas if col in df_busqueda.columns]
    propiedades_filtradas = propiedades[columnas_existentes].to_dict(
        orient='records')

    return jsonify({
        'student_id': student_id,
        'propiedades': propiedades_filtradas
    })

# Ruta para buscar propiedad por ID


@app.route('/buscar_propiedad', methods=['GET'])
def buscar_propiedad_por_id():
    property_id = request.args.get('id', type=int)
    propiedad = df_busqueda[df_busqueda['id'] == property_id]

    if propiedad.empty:
        return jsonify({'error': f"No se encontró la propiedad con ID: {property_id}"}), 404

    detalles_propiedad = propiedad.iloc[0].to_dict()

    return jsonify({
        'id': property_id,
        'detalles': detalles_propiedad
    })


# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
