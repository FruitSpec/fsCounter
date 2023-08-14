import requests
from pandas import DataFrame
from json import load
import os
import sys
from time import sleep
dir_path = os.path.dirname(sys.argv[0])
os.chdir(dir_path)
''' tables = ['fruit_variety', 'project', 'project_plot', 'plot', 'customer','caliber']'''

# DB_SERVER = config['DB']['SERVER']
# DB_USER = config['DB']['USER']
# DB_PASSWORD = config['DB']['PASSWORD']
# DB_NAME = 'fruitspecdb_new_aws'
#
#
# def get_table(SQL_command, params=()):
#     mydb = mysql.connector.connect(
#         host=DB_SERVER,
#         user=DB_USER,
#         password=DB_PASSWORD,
#         database=DB_NAME)
#     cursor = mydb.cursor()
#     cursor.execute(SQL_command, params=params)
#     df = DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])
#     mydb.close()
#     return df

def get_table(SQL_command, config_path ="config.json", params=()):
    with open(config_path, "r", encoding='utf8') as json_file:
        config = load(json_file)
    data = config['DB']['body']
    data['queryString'] = SQL_command % params
    response = requests.post(url=config['DB']['url'], data=data)
    assert response.json()['success']
    df = DataFrame(response.json()['data'])
    return df