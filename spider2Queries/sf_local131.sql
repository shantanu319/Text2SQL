```sql
SELECT 
    MusicalStyle,
    COUNT(CASE WHEN Preference = '1' THEN 1 END) AS FirstPreferenceCount,
    COUNT(CASE WHEN Preference = '2' THEN 1 END) AS SecondPreferenceCount,
    COUNT(CASE WHEN Preference = '3' THEN 1 END) AS ThirdPreferenceCount
FROM 
    MusicPreferences
GROUP BY 
    MusicalStyle;
```