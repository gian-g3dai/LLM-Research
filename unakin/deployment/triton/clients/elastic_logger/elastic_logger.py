import os
import random
import threading
import time

from elasticsearch import Elasticsearch

class ElasticLogger:

    def __init__(self, host, port, user, password):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        
        try:
            # TODO
            cert_path = os.path.join(os.path.dirname(__file__), "elastic_http_ca.crt")
            
            self.elastic_client = Elasticsearch(
                f"https://{host}:{port}",
                ca_certs=cert_path,
                basic_auth=(user, password)
            )
        except Exception as e:
            raise Exception("Elasticsearch connection failed: ", e)

    def log(self, index_name, transaction):
        
        def log_to_elasticsearch(elastic_client, index_name, transaction):

            logger_thread_id = random.randint(0, 10_000_000_000)

            print(f"Logger thread {logger_thread_id} started...")
            time_start_logging = time.time()

            try:
                result = elastic_client.index(index=index_name, body=transaction, refresh=True)

                if result['result'] == 'created':
                    print("Document successfully indexed.")
                else:
                    print("No stats logged for this request. Unexpected result:", result)
            except Exception as e:
                print("No stats logged for this request. Unable to index document:", e)
            
            # Logging finished
            print("Logging finished...")
            time_end_logging = time.time()
            time_logging = time_end_logging - time_start_logging
            print(f"Logger thread {logger_thread_id} finished: time to log {time_logging}")

        thread = threading.Thread(target=log_to_elasticsearch, args=(self.elastic_client, index_name, transaction))
        thread.start()
