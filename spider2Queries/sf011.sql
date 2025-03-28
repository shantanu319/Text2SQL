```sql
SELECT 
    blkgrp_id,
    census_value,
    stcty_tract_id,
    tot_pop,
    (census_value / tot_pop) AS pop_ratio
FROM 
    new_york_state_acs_data
WHERE 
    year = 2021;
```