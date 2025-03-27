```sql
SELECT 
    YEAR(hiring_date) AS year,
    CASE 
        WHEN race = 'Asian' THEN 'Asian'
        WHEN race = 'Black' THEN 'Black'
        WHEN race = 'Hispanic/Latinx' THEN 'Hispanic/Latinx'
        WHEN race = 'White' THEN 'White'
        ELSE 'Other'
    END AS race,
    CASE 
        WHEN gender = 'U.S. Women' THEN 'Women'
        WHEN gender = 'U.S. Men' THEN 'Men'
        ELSE 'Other'
    END AS gender,
    COUNT(*) AS total_hires,
    SUM(CASE WHEN race = 'Asian' THEN 1 ELSE 0 END) AS asian_hires,
    SUM(CASE WHEN race = 'Black' THEN 1 ELSE 0 END) AS black_hires,
    SUM(CASE WHEN race = 'Hispanic/Latinx' THEN 1 ELSE 0 END) AS hispanic_latinx_hires,
    SUM(CASE WHEN race = 'White' THEN 1 ELSE 0 END) AS white_hires,
    SUM(CASE WHEN gender = 'U.S. Women' THEN 1 ELSE 0 END) AS women_hires,
    SUM