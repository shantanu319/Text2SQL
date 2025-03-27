```sql
SELECT AddressNumber, StreetName, StreetType
FROM Addresses
WHERE ZipCodeArea = (SELECT ZipCodeArea FROM Addresses GROUP BY ZipCodeArea ORDER BY Latitude DESC LIMIT 1)
AND State = 'Florida'
ORDER BY Latitude ASC
LIMIT 10;
```