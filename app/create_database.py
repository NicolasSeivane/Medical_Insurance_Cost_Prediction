import psycopg2
from utils import datos, create_table_query, insert_data_from_df, extract_data_to_df, DB_CONFIG
import os

val = 10

if __name__ == "__main__":
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Conexión exitosa a la base de datos PostgreSQL.")
    except Exception as e:
        print(f"❌ Error al conectar a la base de datos: {e}")
        print("\n💡 Asegúrate de que Docker esté corriendo y los contenedores estén activos:")
        print("   docker-compose up -d")
        exit(1)

    try:
        create_table_query(datos, "scoring_dataset", conn, drop_if_exists=True)
        datos = insert_data_from_df(datos, "scoring_dataset", conn, val=val)
        print(f"✅ Tabla 'scoring_dataset' creada e insertados {val} datos aleatorios exitosamente.")
        print("✅ Archivo SQL para la creación de la tabla 'scoring_dataset' generado exitosamente.")
    except Exception as e:
        print(f"❌ Error al crear la tabla o insertar datos: {e}")

    try:
        create_table_query(datos, "training_dataset", conn, drop_if_exists=True)
        datos = insert_data_from_df(datos, "training_dataset", conn)
        print("✅ Tabla 'training_dataset' creada e insertados datos exitosamente.")
        print("✅ Archivo SQL para la creación de la tabla 'training_dataset' generado exitosamente.")
    except Exception as e:
        print(f"❌ Error al crear la tabla o insertar datos: {e}")


    if conn:
        conn.close()
        print("✅ Conexión cerrada exitosamente.")