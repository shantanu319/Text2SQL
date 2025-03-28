```sql
SELECT AVG(TripDuration) AS AverageTripDurationInMinutes
FROM Trips
WHERE TripDate BETWEEN '2016-02-01' AND '2016-02-07'
  AND TripDuration > 0
  AND NumberOfPassengers > 3
  AND TripDistance >= 10
  AND PickupLocation LIKE '%Brooklyn%'
  AND DropoffLocation LIKE '%Brooklyn%';
```