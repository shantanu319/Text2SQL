SELECT v.first_visit_date, t.transaction_date - v.first_visit_date AS days_elapsed, t.device_type