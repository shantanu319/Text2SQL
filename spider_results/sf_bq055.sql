SELECT race, (google_hiring - bls_average) AS difference FROM (
    SELECT race, MAX(google_hiring) AS google_hiring, AVG(bls_hiring) AS bls_average