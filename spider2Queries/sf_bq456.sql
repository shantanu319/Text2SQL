```sql
SELECT 
    d.PatientID,
    d.StudyInstanceUID,
    d.StudyDate,
    f.CodeMeaning AS FindingSiteCodeMeaning,
    MAX(qm.Elongation) AS MaxElongation,
    MAX(qm.Flatness) AS MaxFlatness,
    MAX(qm.LeastAxisIn3DLength) AS MaxLeastAxisIn3DLength,
    MAX(qm.MajorAxisIn3DLength) AS MaxMajorAxisIn3DLength,
    MAX(qm.Maximum3DDiameterOfAMesh) AS MaxMaximum3DDiameterOfAMesh,
    MAX(qm.MinorAxisIn3DLength) AS MaxMinorAxisIn3DLength,
    MAX(qm.Sphericity) AS MaxSphericity,
    MAX(qm.SurfaceAreaOfMesh) AS MaxSurfaceAreaOfMesh,
    MAX(qm.SurfaceToVolumeRatio) AS MaxSurfaceToVolumeRatio,
    MAX(qm.VolumeFromVoxelSummation) AS MaxVolumeFromVoxelSummation,
    MAX(qm.VolumeOfMesh) AS MaxVolumeOfMesh
FROM 
    dicom_all d
JOIN 
    quantitative_measurements qm ON d.segmentationInstanceUID = qm.SOPInstance