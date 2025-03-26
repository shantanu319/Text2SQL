SELECT * FROM olist_geolocation WHERE (geolocation_state, geolocation_city, geolocation_zip_code_prefix, geolocation_lat, geolocation_lng) IN (
  SELECT geolocation_state, geolocation_city, geolocation_zip_code_prefix, geolocation_lat, geolocation_lng
  FROM olist_geolocation
  ORDER BY geolocation_state, geolocation_city, geolocation_zip_code_prefix, geolocation_lat, geolocation_lng