```sql
WITH CPCPatents AS (
    SELECT 
        p.cpc_code,
        p.filing_date,
        p.application_number,
        p.full_cpc_title,
        ROW_NUMBER() OVER (PARTITION BY p.cpc_code ORDER BY p.filing_date) AS rn
    FROM 
        patents p
),
FilteredCPCs AS (
    SELECT 
        cpc_code,
        filing_date,
        application_number,
        full_cpc_title
    FROM 
        CPCPatents
    WHERE 
        rn = 1 AND filing_date IS NOT NULL AND application_number <> ''
),
ExponentialMovingAverage AS (
    SELECT 
        cpc_code,
        YEAR(filing_date) AS filing_year,
        SUM(CASE WHEN rn = 1 THEN 1 ELSE 0 END) / COUNT(*) AS exp_moving_avg
    FROM 
        FilteredCPCs
    GROUP BY 
        cpc_code, YEAR(filing_date)
    WINDOW w AS (ORDER BY YEAR(filing_date) RANGE BETWEEN CURRENT ROW AND 1 FOLLOWING)
)
SELECT 
    e.cpc_code,
    e.filing_year,
    e.exp_moving_avg,
    f.full_cpc_title
FROM 
    ExponentialMovingAverage e