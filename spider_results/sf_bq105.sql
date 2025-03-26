SELECT state, COUNT(*) / (population * 100) AS accidents_per_100k FROM accidents WHERE distracted = 'Yes' AND NOT distracted IS NULL GROUP BY state ORDER BY accidents_per_100k DESC LIMIT 5