SELECT * FROM patents WHERE country = 'United States' AND classification = 'B2' AND grant_date BETWEEN '2010-01-01' AND '2015-01-01' ORDER BY num_forward_citations DESC LIMIT 1