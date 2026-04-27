import docker
import time
from elasticsearch import Elasticsearch
from datetime import datetime

def monitor_containers():
    """
    Monitore les conteneurs Docker et envoie les métriques vers Elasticsearch.
    """
    es = Elasticsearch("http://localhost:9200")
    client = docker.from_env()

    print("Démarrage du monitoring des conteneurs Docker...")

    containers = client.containers.list()
    for container in containers:
        stats = container.stats(stream=False)
        
        # CPU
        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                    stats["precpu_stats"]["cpu_usage"]["total_usage"]
        system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                       stats["precpu_stats"]["system_cpu_usage"]
        cpu_percent = (cpu_delta / system_delta) * 100.0

        # Mémoire
        mem_usage = stats["memory_stats"]["usage"]
        mem_limit = stats["memory_stats"]["limit"]
        mem_percent = (mem_usage / mem_limit) * 100.0

        doc = {
            "timestamp": datetime.now().isoformat(),
            "container_name": container.name,
            "cpu_percent": round(cpu_percent, 2),
            "memory_usage_mb": round(mem_usage / 1024 / 1024, 2),
            "memory_percent": round(mem_percent, 2),
            "status": container.status
        }

        es.index(index="docker-monitoring", body=doc)
        print(f"Container {container.name}: CPU={cpu_percent:.2f}% MEM={mem_percent:.2f}%")

    print("Métriques Docker envoyées vers Elasticsearch ✅")

if __name__ == "__main__":
    monitor_containers()