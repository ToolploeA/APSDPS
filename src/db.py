import sqlite3, pprint, os, time

# .header on
# .mode column

class DB():
    def __init__(self, db_file, data):
        new = not os.path.exists(db_file)
        self.db = sqlite3.connect(db_file)
        self.cursor = self.db.cursor()
        if new:
            self.cursor.execute('''
            CREATE TABLE resource (
                node char(4) DEFAULT NULL,
                resource_now tinyint DEFAULT NULL,
                active tinyint DEFAULT NULL
                )
            ''')
            self.cursor.execute('''
            CREATE TABLE job (
                job_id int DEFAULT NULL,
                node char(4) DEFAULT NULL
                )
            ''')
            self.db.commit()
        self.import_data(data)
    
    def import_data(self, data):
        # Clear existing data
        self.cursor.execute('DELETE FROM resource')
        # insert new data
        for node, info in data.items():
            insert_sql = f'''
            INSERT INTO resource (node, resource_now, active)
            VALUES ("{node}", {info['calc_resource']}, {info['active']})
            '''
            self.cursor.execute(insert_sql)
            self.db.commit()

    def get_resource(self, resource):
        result = None
        select_sql = f'SELECT node FROM resource WHERE resource_now >= {resource} AND active = 1 ORDER BY resource_now ASC LIMIT 1'
        while 1:
            self.cursor.execute(select_sql)
            result = self.cursor.fetchone() # (node: str, ) or None
            if not result:
                time.sleep(1)
            else:
                break
        node = result[0]
        update_sql = f'UPDATE resource SET resource_now = resource_now - {resource} WHERE node = "{node}"'
        self.cursor.execute(update_sql)
        self.db.commit()
        return node
    
    def release_resource(self, node, resource):
        update_sql = f'UPDATE resource SET resource_now = resource_now + {resource} WHERE node = "{node}"'
        self.cursor.execute(update_sql)
        self.db.commit()

    def add_job(self, node, job_id):
        insert_sql = f'INSERT INTO job (job_id, node) VALUES ({job_id}, "{node}")'
        self.cursor.execute(insert_sql)
        self.db.commit()

    def remove_job(self, job_id):
        delete_sql = f'DELETE FROM job WHERE job_id = {job_id}'
        self.cursor.execute(delete_sql)
        self.db.commit()

    def check_done(self):
        select_sql = 'SELECT CASE WHEN EXISTS (SELECT 1 from job) THEN 0 ELSE 1 END AS is_empty'
        self.cursor.execute(select_sql)
        result = self.cursor.fetchone()
        return bool(result[0])

    def get_resource_pool_size(self):
        select_sql = 'SELECT SUM(resource_now) AS resource_pool_size FROM resource WHERE active = 1'
        self.cursor.execute(select_sql)
        result = self.cursor.fetchone()
        return int(result[0])

    def __repr__(self):
        select_sql = 'SELECT * FROM resource'
        self.cursor.execute(select_sql)
        resource_result = self.cursor.fetchall()
        select_sql = 'SELECT * FROM job'
        self.cursor.execute(select_sql)
        job_result = self.cursor.fetchall()
        return pprint.pformat(resource_result) + '\n' + pprint.pformat(job_result)

'''
    def active_node(self, node):
        update_sql = f'UPDATE resource SET active = 1 WHERE node = "{node}"'
        self.cursor.execute(update_sql)
        self.db.commit()

    def activate_all_nodes(self):
        update_sql = 'UPDATE resource SET active = 1'
        self.cursor.execute(update_sql)
        self.db.commit()

    def deactive_node(self, node):
        update_sql = f'UPDATE resource SET active = 0 WHERE node = "{node}"'
        self.cursor.execute(update_sql)
        self.db.commit()

    def deactive_all_nodes(self):
        update_sql = 'UPDATE resource SET active = 0'
        self.cursor.execute(update_sql)
        self.db.commit()

    def pause(self):
        insert_sql = 'INSERT INTO job (job_id) VALUES (999999)'
        self.cursor.execute(insert_sql)
        self.db.commit()

    def unpause(self):
        delete_sql = 'DELETE FROM job WHERE job_id = 999999'
        self.cursor.execute(delete_sql)
        self.db.commit()

    def modify_cores(self, node, num_cores_total):
        update_sql = f'UPDATE resource SET num_cores_total = {num_cores_total}, num_cores_avail = {num_cores_total} WHERE node = "{node}"'
        self.cursor.execute(update_sql)
        self.db.commit()
    
    def modify_orders(self, node, order):
        update_sql = f'UPDATE resource SET use_order = {order} WHERE node = "{node}"'
        self.cursor.execute(update_sql)
        self.db.commit()
    
    def set_orders(self, node_list, order_list):
        for node, order in zip(node_list, order_list):
            self.modify_orders(node, order)
'''