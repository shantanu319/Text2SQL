```sql
SELECT 
    DATE,
    Temperature,
    Precipitation,
    WindSpeed,
    AVG(Temperature) OVER (ORDER BY DATE ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS MovingAverageTemperature,
    AVG(Precipitation) OVER (ORDER BY DATE ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS MovingAveragePrecipitation,
    AVG(WindSpeed) OVER (ORDER BY DATE ROWS BETWEEN 7 PRECEDING AND CURRENT ROW) AS MovingAverageWindSpeed,
    MovingAverageTemperature - LAG(MovingAverageTemperature, 1) OVER (ORDER BY DATE) AS Lag1Difference,
    MovingAverageTemperature - LAG(MovingAverageTemperature, 2) OVER (ORDER BY DATE) AS Lag2Difference,
    MovingAverageTemperature - LAG(MovingAverageTemperature, 3) OVER (ORDER BY DATE) AS Lag3Difference,
    MovingAverageTemperature - LAG(MovingAverageTemperature, 4) OVER (ORDER BY DATE) AS Lag4Difference,
    MovingAverageTemperature - LAG(MovingAverageTemperature, 5) OVER (ORDER BY DATE) AS Lag5Difference,
    MovingAverageTemperature - LAG(MovingAverageTemperature, 6) OVER (ORDER BY