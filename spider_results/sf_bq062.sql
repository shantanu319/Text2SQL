```sql
SELECT 
    sys.name AS System,
    lic.license_name AS License,
    COUNT(*) AS Frequency
FROM 
    package_license pl
JOIN 
    system sys ON pl.system_id = sys.id
JOIN 
    license lic ON pl.license_id = lic.id
GROUP BY 
    sys.name, lic.license_name
ORDER BY 
    Frequency DESC;
```