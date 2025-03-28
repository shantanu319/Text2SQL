```sql
SELECT 
    stat.state_name,
    SUM(CASE WHEN crop.crop_name = 'corn' THEN value ELSE 0 END) AS corn_production,
    SUM(CASE WHEN crop.crop_name = 'mushrooms' THEN value ELSE 0 END) AS mushroom_production
FROM 
    crops
JOIN 
    states stat ON crops.stat_id = stat.stat_id
JOIN 
    crops_categories cat ON crops.cat_id = cat.cat_id
JOIN 
    crops_units unit ON crops.unit_id = unit.unit_id
JOIN 
    crops_crops crop ON crops.crop_id = crop.crop_id
WHERE 
    stat.statisticcat_desc = 'PRODUCTION'
    AND stat.agg_level_desc = 'STATE'
    AND crops.value IS NOT NULL
    AND cat.category_name = 'FIELD CROPS' FOR corn
    AND cat.category_name = 'HORTICULTURE' FOR mushrooms
    AND unit.unit_desc = 'BU'
GROUP BY 
    stat.state_name;
```