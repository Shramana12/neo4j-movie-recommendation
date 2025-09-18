CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.title IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Actor) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Director) REQUIRE d.name IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///movies.csv' AS row
MERGE (m:Movie {title: row.title, genre: row.genre})
MERGE (a:Actor {name: row.actor})
MERGE (d:Director {name: row.director})
MERGE (a)-[:ACTED_IN]->(m)
MERGE (d)-[:DIRECTED]->(m);
