```sql
SELECT p.name
FROM players p
JOIN matches m ON p.match_id = m.id
WHERE m.winner_team != p.team AND p.runs >= 100;
```