```sql
SELECT 
    date_created,
    date_started,
    reactions_array,
    outcomes_array,
    products_brand_name_cleaned,
    products_industry_code_cleaned,
    products_role_cleaned,
    products_industry_name_cleaned,
    industry_code_length,
    brand_name_length
FROM 
    food_events_data
WHERE 
    date_created BETWEEN '2015-01-01' AND '2015-01-31'
    AND date_started BETWEEN '2015-01-01' AND '2015-01-31'
    AND reactions LIKE '%,%'
    AND outcomes LIKE '%,%'
    AND products_brand_name LIKE '%,%'
    AND products_industry_code LIKE '%,%'
    AND products_role LIKE '%,%'
    AND products_industry_name LIKE '%,%'
    AND REPLACE(REPLACE(products_brand_name, ', ', ' -- '), ',', ' -- ') = products_brand_name_cleaned
    AND REPLACE(REPLACE(products_industry_code, ', ', ' -- '), ',', ' -- ') = products_industry_code_cleaned
    AND REPLACE(REPLACE(products_role, ', ', ' -- '), ',', ' -- ') = products_role_cleaned
    AND