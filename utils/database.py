import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """
    Cria e retorna uma conexão com o banco de dados PostgreSQL usando as credenciais definidas no .env.
    """
    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT")
        )
        return connection
    except Exception as e:
        print("Erro ao conectar com o banco de dados:", e)
        return None


def execute_query(query, params=None):
    """
    Executa uma query SQL e retorna o resultado, se aplicável.

    Se a query for um SELECT, retorna os registros como lista de dicionários.
    Para INSERT, UPDATE ou DELETE, executa a operação e realiza o commit.

    Args:
        query (str): A query SQL a ser executada.
        params (tuple, opcional): Parâmetros para a query.

    Returns:
        list[dict] ou None: Resultados para SELECT ou None para operações de modificação.
    """
    connection = get_connection()
    if connection is None:
        return None

    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, params)

        # Se for uma consulta SELECT, busca os resultados
        if query.strip().lower().startswith("select"):
            result = cursor.fetchall()
        else:
            connection.commit()
            result = None

        cursor.close()
        connection.close()
        return result
    except Exception as e:
        print("Erro ao executar a query:", e)
        if connection:
            connection.rollback()
            connection.close()
        return None
