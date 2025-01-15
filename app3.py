import sqlite3
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Conectar a la base de datos SQLite o crear una nueva si no existe
def conectar_db(nombre_db):
    try:
        conexion = sqlite3.connect(nombre_db)
        print(f"Conexión exitosa a la base de datos {nombre_db}")
        return conexion
    except sqlite3.Error as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

# Crear una tabla en la base de datos para almacenar los datos y embeddings
def crear_tabla(conexion):
    try:
        cursor = conexion.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documentos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                texto TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        ''')
        conexion.commit()
        print("Tabla 'documentos' creada exitosamente")
    except sqlite3.Error as e:
        print(f"Error al crear la tabla: {e}")

# Insertar un nuevo documento en la tabla
def insertar_documento(conexion, texto, embedding):
    try:
        cursor = conexion.cursor()
        cursor.execute('''
            INSERT INTO documentos (texto, embedding)
            VALUES (?, ?)
        ''', (texto, embedding))
        conexion.commit()
        print("Documento insertado exitosamente")
    except sqlite3.Error as e:
        print(f"Error al insertar el documento: {e}")

# Leer todos los documentos de la tabla
def leer_documentos(conexion):
    try:
        cursor = conexion.cursor()
        cursor.execute('SELECT * FROM documentos')
        documentos = cursor.fetchall()
        for documento in documentos:
            print(documento)
    except sqlite3.Error as e:
        print(f"Error al leer los documentos: {e}")

# Actualizar un documento en la tabla
def actualizar_documento(conexion, id, texto, embedding):
    try:
        cursor = conexion.cursor()
        cursor.execute('''
            UPDATE documentos
            SET texto = ?, embedding = ?
            WHERE id = ?
        ''', (texto, embedding, id))
        conexion.commit()
        print("Documento actualizado exitosamente")
    except sqlite3.Error as e:
        print(f"Error al actualizar el documento: {e}")

# Eliminar un documento de la tabla
def eliminar_documento(conexion, id):
    try:
        cursor = conexion.cursor()
        cursor.execute('DELETE FROM documentos WHERE id = ?', (id,))
        conexion.commit()
        print("Documento eliminado exitosamente")
    except sqlite3.Error as e:
        print(f"Error al eliminar el documento: {e}")

# Crear embeddings utilizando TF-IDF
def crear_embeddings(textos):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(textos).toarray()
    return embeddings

# Convertir embeddings a formato binario para almacenar en la base de datos
def convertir_a_binario(embedding):
    return sqlite3.Binary(embedding.tobytes())

# Convertir binario a embeddings
def convertir_a_embedding(binario):
    return np.frombuffer(binario, dtype=np.float64)

# Nombre de la base de datos
nombre_db = 'data/embeddings.db'

# Asegurarse de que el directorio exista
os.makedirs(os.path.dirname(nombre_db), exist_ok=True)

# Conectar a la base de datos
conexion = conectar_db(nombre_db)

# Crear la tabla
crear_tabla(conexion)

# Insertar un nuevo documento
texto = "Este es un documento de ejemplo."
embedding = crear_embeddings([texto])[0]
print("Embedding generado:", embedding)
insertar_documento(conexion, texto, convertir_a_binario(embedding))

# Leer todos los documentos
leer_documentos(conexion)

# Actualizar un documento
nuevo_texto = "Este es un documento actualizado."
nuevo_embedding = crear_embeddings([nuevo_texto])[0]
print("Nuevo embedding generado:", nuevo_embedding)
actualizar_documento(conexion, 1, nuevo_texto, convertir_a_binario(nuevo_embedding))

# Leer todos los documentos nuevamente
leer_documentos(conexion)

# Eliminar un documento
eliminar_documento(conexion, 1)

# Leer todos los documentos nuevamente
leer_documentos(conexion)

# Cerrar la conexión
conexion.close()