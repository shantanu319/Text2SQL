```sql
SELECT 
    m1.molecule_id AS molecule_id_1,
    m2.molecule_id AS molecule_id_2,
    MAX(m1.heavy_atoms) AS max_heavy_atoms,
    DATE_FORMAT(
        CONCAT_WS('-', 
            YEAR(p.publication_date),
            LPAD(MONTH(p.publication_date), 2, '0'),
            LPAD(DAY(p.publication_date), 2, '0')
        ), '%Y-%m-%d'
    ) AS latest_publication_date,
    MAX(d.document_id) AS highest_document_id,
    CASE 
        WHEN s.standard_value > s.previous_standard_value THEN 'increase'
        WHEN s.standard_value < s.previous_standard_value THEN 'decrease'
        ELSE 'no-change'
    END AS standard_change,
    UUID() AS uuid
FROM 
    assays a
JOIN 
    assays_molecules am ON a.assay_id = am.assay_id
JOIN 
    molecules m1 ON am.molecule_id = m1.molecule_id
JOIN 
    assays_molecules am2 ON a.assay_id = am2.assay_id
JOIN 
    molecules m2 ON am2.molecule_id = m2.molecule_id
LEFT