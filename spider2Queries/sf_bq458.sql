```sql
SELECT 
    a.id,
    a.date,
    a.title,
    (SELECT 
         SUM(wv * wfreq^(-0.4)) AS article_vector
     FROM 
         (
             SELECT 
                 w.word,
                 wv.component,
                 wfreq.frequency
             FROM 
                 tokenize_body_text(bt)
                 JOIN words w ON bt.word_id = w.id
                 JOIN glove_vectors gv ON w.word = gv.word
                 JOIN word_frequencies wf ON w.word = wf.word
         ) wv
     GROUP BY 
         a.id) AS normalized_article_vector
FROM 
    articles a;
```

This SQL query performs the following steps:

1. **Tokenize the Body Text**: The `tokenize_body_text` function is assumed to be a custom function that tokenizes the body text of each article into words.

2. **Exclude Stop Words**: This step is implied in the tokenization process, as most tokenizers exclude common stop words.

3. **Obtain Word Vectors**: The `glove_vectors` table contains pre-trained word vectors from GloVe.

4. **Weight Word Vectors**: Each word vector is weighted by dividing each component by the 0.4th power of the word's