```sql
SELECT 
    T1.pub_num,
    T2.app_num,
    T3.claim_num,
    T4.word_count
FROM 
    uspto_oce_claims AS T1
JOIN 
    uspto_oce_claims_match AS T2 ON T1.match_id = T2.match_id
JOIN 
    patents_publications AS T3 ON T1.pub_num = T3.pub_num
JOIN 
    (
        SELECT 
            app_num,
            MAX(word_count) AS max_word_count
        FROM 
            uspto_oce_claims
        WHERE 
            ind_flg = '1'
        GROUP BY 
            app_num
    ) AS T4 ON T1.app_num = T4.app_num AND T1.word_count = T4.max_word_count
WHERE 
    T1.ind_flg = '1'
ORDER BY 
    T4.word_count DESC
LIMIT 100;
```