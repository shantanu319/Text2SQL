```sql
SELECT 
    SeriesInstanceUID,
    SeriesNumber,
    PatientID,
    StudyInstanceUID,
    Modality,
    SeriesSizeInMiB,
    ImageType,
    TransferSyntaxUID,
    SliceInterval,
    ExposureLevel,
    ImageOrientationPatient,
    PixelSpacing,
    ImagePositionPatient,
    PixelDimensions,
    NumberOfImages,
    UniqueZAxisPositions
FROM 
    CT_Series
WHERE 
    Modality = 'CT'
    AND Collection != 'nlst'
    AND ImageType NOT IN ('LOCALIZER')
    AND TransferSyntaxUID NOT IN ('1.2.840.10008.1.2.4.70', '1.2.840.10008.1.2.4.51')
    AND SliceInterval IS NOT NULL
    AND ExposureLevel IS NOT NULL
    AND ImageOrientationPatient IS NOT NULL
    AND PixelSpacing IS NOT NULL
    AND ImagePositionPatient IS NOT NULL
    AND PixelDimensions IS NOT NULL
    AND NumberOfImages = COUNT(DISTINCT ZAxisPosition)
    AND ABS(CROSS_PRODUCT_XY_DIRECTION_COSINES) BETWEEN 0.99 AND 1.01