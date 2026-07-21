import urllib.request
import json

try:
    with urllib.request.urlopen("http://localhost:6333/collections") as response:
        print("Qdrant Collections:", response.read().decode('utf-8'))
except Exception as e:
    print("Qdrant error:", e)

try:
    with urllib.request.urlopen("http://localhost:9200/_cat/indices?v") as response:
        print("Elasticsearch Indices:\n", response.read().decode('utf-8'))
except Exception as e:
    print("Elasticsearch error:", e)
