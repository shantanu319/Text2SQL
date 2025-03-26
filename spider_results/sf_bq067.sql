CREATE TABLE labeled_dataset (
    accident_id INT PRIMARY KEY,
    state_number VARCHAR(2),
    vehicle_body_type VARCHAR(30),
    num_drunk_drivers INT,
    day_of_week VARCHAR(10),
    hour_of_crash INT,
    work_zone_indicator BOOLEAN,
    avg_abs_diff_travel_speed_speed_limit DECIMAL(5,2)
);