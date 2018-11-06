# connect sql
import pymysql

mysql_config = {'host':'192.168.2.6',
                'user':'kevinlin_finance',
                'passwd':'gP1dVymsEc',
                'db':'kevinlin_finance',
                }
con = pymysql.connect(host=mysql_config['host'], user=mysql_config['user'],
                    passwd=mysql_config['passwd'], db=mysql_config['db'], charset="utf8")
cur = con.cursor()
query = u"CREATE TABLE table_name"
cur.execute(query)
con.commit()
con.close()

## query
# create table
query = u"CREATE TABLE table_name"

# drop table
query = u"DROP TABLE IF EXISTS table_name"

# insert table
query = u"INSERT INTO table_name VALUES({})".format(data)

# create table
datatype = [("id", "VARCHAR(100)"), 
            ("rank", "INT"), 
            ("price_usd", "DOUBLE"),
            ("percent_change_7d", "FLOAT"),
            ("last_updated", "DATETIME")]
def create_table(datatypes):
    con = pymysql.connect(host=mysql_config['host'], user=mysql_config['user'],
                        passwd=mysql_config['passwd'], db=mysql_config['db'], charset="utf8")
    cur = con.cursor()
    columns = ", ".join([" ".join(d) for d in datatypes])
    cur.execute(u"""DROP TABLE IF EXISTS table_name""")
    cur.execute(u"""CREATE TABLE table_name ({})""".format(columns))
    con.commit()
    con.close()

# insert dataframe into datebase
def save_in_database(table_name, df):
    con = pymysql.connect(host=mysql_config['host'], user=mysql_config['user'], passwd=mysql_config['passwd'],
                          db=mysql_config['db'], charset="utf8")
    cur = con.cursor()
    ncol = df.shape[1]
    num_values = ",".join(['%s']*(ncol))
    df = df.values.tolist()
    query = '''INSERT INTO {} VALUES({})'''.format(table_name, num_values)
    cur.executemany(query, df) 
    con.commit()
    con.close()

# add a new column


# unique row
query = u"CREATE TABLE table_2 LIKE table"
query = u"ALTER TABLE table_2 ADD UNIQUE INDEX unique_index(datetime, symbol1, symbol2)"
query = u"INSERT IGNORE table_2 SELECT * FROM table"
