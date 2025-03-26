SELECT AVG(series_size_in_MiB)
FROM (
    SELECT patient_id, SUM(instance_size_in_MiB) AS series_size_in_MiB
    FROM (
        SELECT patient_id, image_id, MAX(slice_interval_difference_tolerance) AS max_slice_interval_difference_tolerance,
               MIN(slice_interval_difference_tolerance) AS min_slice_interval_difference_tolerance,
               MAX(exposure_difference) AS max_exposure_difference,
               MIN(exposure_difference) AS min_exposure_difference
        FROM (
            SELECT p.patient_id, i.image_id, s.slice_interval_difference_tolerance, e.exposure_difference
            FROM nlst.pacs_images i
            JOIN nlst.pacs_patients p ON i.patient_id = p.patient_id
            JOIN nlst.pacs_series s ON i.series_id = s.series_id
            JOIN nlst.pacs_exposures e ON i.exposure_id = e.exposure_id
            WHERE s.collection_name = 'nlst'
        ) subquery
        GROUP BY patient_id
    ) subquery2
    ORDER BY max_slice_interval_difference_tolerance DESC, max_exposure_difference DESC
    LIMIT 3