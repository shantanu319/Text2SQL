```sql
WITH ArticleVectors AS (
    SELECT 
        id,
        date,
        title,
        body,
        -- Tokenize body without stopwords
        -- Convert tokens to GloVe-based word vectors
        -- Weight each vector by dividing each dimension by the 0.4th power of its word frequency
        -- Sum and normalize the weighted vectors to get a unit vector for each article
        -- Compute cosine similarity between the query vector and each article vector
        -- Return the id, date, title, and cosine similarity score for the top 10 articles with the highest similarity
        -- ...
),
QueryVector AS (
    -- Similar process as above for the query phrase
    -- ...
)
SELECT 
    av.id,
    av.date,
    av.title,
    COSINE_SIMILARITY(av.vector, q.vector) AS similarity_score
FROM 
    ArticleVectors av
JOIN 
    QueryVector q ON true
ORDER BY 
    similarity_score DESC
LIMIT 10;
```

Note: The actual implementation of tokenization, GloVe conversion, weighting, normalization, and cosine similarity computation would require additional steps and functions that are not included in this simplified example. Additionally, the `COSINE_SIMILARITY` function would