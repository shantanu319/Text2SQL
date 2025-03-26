SELECT f.family_id, a.abstract 
FROM publications p JOIN abs_and_emb ae ON p.publication_number = ae.publication_number