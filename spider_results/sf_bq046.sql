SELECT barcode, gdc_file_url FROM cases WHERE gender = 'female' AND age <= 30 AND diagnosis = 'breast cancer' AND clinical_history LIKE '%problematic%prior%treatments%' OR