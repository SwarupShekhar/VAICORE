import os
import redis
from rq import Worker, Queue

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)

if __name__ == '__main__':
    print("Starting RQ Worker...")
    worker = Worker(['default'], connection=conn)
    worker.work()
