SELECT AVG(anxiety_level) AS avg_anxiety_2019, AVG(depression_level) AS avg_depression_2019,
       AVG(anxiety_level - avg_anxiety_2019) * 100 / (AVG(anxiety_level) - avg_anxiety_2019) AS anxiety_increase_percentage,
       AVG(depression_level - avg_depression_2019) * 100 / (AVG(depression_level) - avg_depression_2019) AS depression_increase_percentage