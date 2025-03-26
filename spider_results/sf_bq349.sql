SELECT osm_id FROM planet_features WHERE type = 'multipolygon' AND EXISTS (
    SELECT 1 FROM planet_nodes WHERE tags LIKE '%amenity=%'
)