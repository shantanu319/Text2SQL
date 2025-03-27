```sql
SELECT 
    b.bowler_name,
    m.match_number,
    g.game_number,
    h.handicap_score,
    t.tournament_date,
    t.location
FROM 
    bowler b
JOIN 
    match m ON b.bowler_id = m.bowler_id
JOIN 
    game g ON m.match_id = g.match_id
JOIN 
    handicap h ON g.game_id = h.game_id
JOIN 
    tournament t ON g.tournament_id = t.tournament_id
WHERE 
    h.handicap_score <= 190
    AND t.location IN ('Thunderbird Lanes', 'Totem Lanes', 'Bolero Lanes')
    AND g.winner = b.bowler_id;
```