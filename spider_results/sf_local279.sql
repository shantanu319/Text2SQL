SELECT p.product_id, m.month, ABS(m.ending_inventory - p.minimum_required_level) AS diff