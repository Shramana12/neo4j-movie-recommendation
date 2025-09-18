import os
from neo4j import GraphDatabase

URI   = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER  = os.getenv("NEO4J_USER", "neo4j")
PASS  = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(URI, auth=(USER, PASS))

def find_movies_by_actor(actor_name):
    q = """
    MATCH (a:Actor {name: $actor})-[:ACTED_IN]->(m:Movie)
    RETURN m.title AS movie
    """
    with driver.session() as session:
        res = session.run(q, actor=actor_name)
        return [r["movie"] for r in res]

if __name__ == "__main__":
    print("Movies by Leonardo DiCaprio:")
    print(find_movies_by_actor("Leonardo DiCaprio"))
