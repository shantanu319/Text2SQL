SELECT * FROM dicom_all WHERE not (collection = 'NLST') AND transfer_syntax NOT IN ('1.2.840.10008.1.2.4.70', '1.2.840.10008.1.2.4.51')