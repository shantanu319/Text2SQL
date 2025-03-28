```sql
SELECT 
    YEAR(LOSS_DATE) AS Loss_Year,
    SUM(BUILDING_DAMAGE_AMOUNT) AS Total_Building_Damage,
    SUM(CONTENTS_DAMAGE_AMOUNT) AS Total_Contents_Damage
FROM 
    nfip_losses
WHERE 
    COMMUNITY_NAME = 'City Of New York'
    AND LOSS_DATE BETWEEN '2010-01-01' AND '2019-12-31'
GROUP BY 
    YEAR(LOSS_DATE);
```