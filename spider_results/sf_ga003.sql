```sql
SELECT 
    board_type,
    AVG(score) AS average_score
FROM 
    quick_play_mode_completions
WHERE 
    completion_date = '2018-09-15'
GROUP BY 
    board_type;
```