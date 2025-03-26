SELECT AVG(CASE WHEN description LIKE '%discharge%' AND NOT description IN ('Urethral discharge', 'Discharge of firework', 'Legal intervention involving firearm discharge') THEN 1 ELSE 0 END) AS avg_discharge_deaths,
       AVG(CASE WHEN description LIKE '%vehicle%' THEN 1 ELSE 0 END) AS avg_vehicle_deaths