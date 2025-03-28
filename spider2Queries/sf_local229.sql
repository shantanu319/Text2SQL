To achieve the desired result using SQL, you need to join the `players` table with itself on the `match_id` column to find partnerships between players. Then, calculate the total partnership score for each match and identify the top two players based on their individual scores. Here's how you can write the SQL query:

```sql
WITH PlayerPartnerships AS (
    SELECT 
        p1.id AS player1_id,
        p2.id AS player2_id,
        SUM(p1.runs + p2.runs) AS total_partnership_score
    FROM 
        players p1
    JOIN 
        players p2 ON p1.match_id = p2.match_id AND p1.id < p2.id
    GROUP BY 
        p1.id, p2.id
),
RankedPlayers AS (
    SELECT 
        player1_id,
        player2_id,
        total_partnership_score,
        ROW_NUMBER() OVER (PARTITION BY total_partnership_score ORDER BY player1_id DESC) AS rn1,
        ROW_NUMBER() OVER (PARTITION BY total_partnership_score ORDER BY player2_id ASC) AS rn2
    FROM 
        PlayerPartnerships
)
SELECT 
    player1_id,
    player2