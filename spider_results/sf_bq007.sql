SELECT state_name, SUM(weighted_employment) AS total_weighted_employment FROM (
    SELECT state_name, SUM(employment_sector * weight) AS weighted_employment
    FROM (
        SELECT state_name, sector, SUM(value) AS employment_sector
        FROM (
            SELECT state_name, 'Employment in Agriculture' AS sector, value
            FROM acs_data_2017 WHERE year = 2017 AND variable = 'B19013'
            UNION ALL
            SELECT state_name, 'Employment in Manufacturing' AS sector, value
            FROM acs_data_2017 WHERE year = 2017 AND variable = 'B19014'
            UNION ALL
            SELECT state_name, 'Employment in Construction' AS sector, value
            FROM acs_data_2017 WHERE year = 2017 AND variable = 'B19015'
            UNION ALL
            SELECT state_name, 'Employment in Transportation and Warehousing' AS sector, value
            FROM acs_data_2017 WHERE year = 2017 AND variable = 'B19016'
            UNION ALL
            SELECT state_name, 'Employment in Information' AS sector, value
            FROM acs_data_2017 WHERE year = 2017 AND variable = 'B19017'
            UNION ALL
            SELECT state_name, 'Employment in Finance and Insurance' AS sector, value
            FROM acs_data_2017 WHERE year = 2017 AND variable = 'B19018'
            UNION ALL
            SELECT state_name, 'Employment in Real Estate and Rental and Leasing' AS sector, value
            FROM acs_data_2017 WHERE year = 2017 AND variable = 'B19019'
            UNION ALL
            SELECT state_name, 'Employment in Professional, Scientific, and Technical Services' AS sector, value
            FROM acs_data_2017 WHERE year = 2017 AND variable = 'B19020'
            UNION ALL
            SELECT state_name, 'Employment in Management, Business, and Financial Operations' AS sector, value
            FROM acs_data_2017 WHERE year = 2017 AND variable = 'B190