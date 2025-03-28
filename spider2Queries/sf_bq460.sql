```sql
WITH TokenizedArticles AS (
    SELECT 
        a.id,
        a.date,
        a.title,
        SUBSTRING_INDEX(SUBSTRING_INDEX(a.body_text, ' ', n), ' ', -1) AS word
    FROM 
        nature_articles a
    CROSS JOIN 
        (SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5) n
    WHERE 
        n.n <= LENGTH(a.body_text) - LENGTH(REPLACE(a.body_text, ' ', '')) + 1
),
FilteredWords AS (
    SELECT 
        ta.id,
        ta.word
    FROM 
        TokenizedArticles ta
    LEFT JOIN 
        stopwords s ON ta.word = s.stopword
    WHERE 
        s.stopword IS NULL
),
WordVectors AS (
    SELECT 
        fw.id,
        fw.word,
        gv.vector
    FROM 
        FilteredWords fw
    INNER JOIN 
        glove_vectors gv ON fw.word = gv.word
),
WordFrequencies AS (
    SELECT 
        fw.id,
        fw.word,
        wf.frequency
    FROM 
        FilteredWords fw
    INNER JOIN 
        word_frequencies wf ON fw.word = wf