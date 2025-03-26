SELECT p.name, MAX(p.pitch_speed) 
FROM players p JOIN games g ON p.id = g.p_id