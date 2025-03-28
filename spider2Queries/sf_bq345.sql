```sql
SELECT 
    c.collection_id,
    s.study_id,
    s.series_id,
    SUM(f.file_size) AS total_file_size_kb
FROM 
    file f
JOIN 
    dicom_image di ON f.id = di.file_id
JOIN 
    collection c ON di.collection_id = c.id
JOIN 
    study s ON di.study_id = s.id
WHERE 
    di.modalities LIKE '%SEG%' OR di.modalities LIKE '%RTSTRUCT%'
    AND di.sop_class_uid = '1.2.840.10008.5.1.4.1.1.66.4'
    AND NOT EXISTS (
        SELECT 1
        FROM reference r
        WHERE r.source_id = f.id
    )
GROUP BY 
    c.collection_id, s.study_id, s.series_id
ORDER BY 
    total_file_size_kb DESC;

-- Viewer URLs
SELECT 
    CONCAT('https://viewer.imaging.datacommons.cancer.gov/viewer/', s.study_id) AS viewer_url,
    s.study_id,
    SUM(f.file_size) AS total_file_size_kb
FROM 
    file f
JOIN 
    dicom_image di ON f.id = di.file_id