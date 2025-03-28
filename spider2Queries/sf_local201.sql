```sql
SELECT 
    word,
    COUNT(*) AS anagram_count
FROM (
    SELECT 
        word,
        LENGTH(word) AS word_length,
        GROUP_CONCAT(DISTINCT SUBSTRING_INDEX(SUBSTRING_INDEX(word, '', n), '', -1)) AS anagrams
    FROM (
        SELECT 
            word,
            LENGTH(word) AS word_length,
            ROW_NUMBER() OVER (PARTITION BY LENGTH(word) ORDER BY word ASC) AS rn
        FROM (
            SELECT 
                word
            FROM (
                SELECT 
                    word
                FROM your_table_name
                WHERE 
                    LENGTH(word) BETWEEN 4 AND 5
                    AND word LIKE 'r%'
                    AND word NOT IN (
                        SELECT 
                            word
                        FROM your_table_name
                        EXCEPT
                        SELECT 
                            word
                        FROM your_table_name
                        WHERE 
                            word = word
                    )
            ) AS filtered_words
            ORDER BY word ASC
            LIMIT 10
        ) AS ranked_words
    ) AS word_with_r
    JOIN (
        SELECT 
            word_length,
            GROUP_CONCAT(anagrams SEPARATOR ',') AS anagrams_list
        FROM (
            SELECT 
                word_length,
                anagrams