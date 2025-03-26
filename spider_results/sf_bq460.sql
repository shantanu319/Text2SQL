SELECT a.id, a.date, a.title, c.cosine_similarity 
FROM articles AS a 
JOIN (