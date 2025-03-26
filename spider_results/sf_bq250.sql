SELECT SUM(population) FROM (
    SELECT ST_ConvexHull(ST_Collect(centroid)) AS bounding_region,
           ST_Distance(ST_ClosestPoint(bounding_region, centroid), centroid) AS distance_to_hospital
    FROM (
        SELECT ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) AS centroid,
               population
        FROM (
            SELECT longitude, latitude, population
            FROM (
                SELECT longitude, latitude, population
                FROM (
                    SELECT longitude, latitude, population
                    FROM (
                        SELECT longitude, latitude, population
                        FROM (
                            SELECT longitude, latitude, population
                            FROM (
                                SELECT longitude, latitude, population
                                FROM (
                                    SELECT longitude, latitude, population
                                    FROM (
                                        SELECT longitude, latitude, population
                                        FROM (
                                            SELECT longitude, latitude, population
                                            FROM (
                                                SELECT longitude, latitude, population
                                                FROM (
                                                    SELECT longitude, latitude, population
                                                    FROM (
                                                        SELECT longitude, latitude, population
                                                        FROM (
                                                            SELECT longitude, latitude, population
                                                            FROM (
                                                                SELECT longitude, latitude, population
                                                                FROM (
                                                                    SELECT longitude, latitude, population
                                                                    FROM (
                                                                        SELECT longitude, latitude, population
                                                                        FROM (
                                                                            SELECT longitude, latitude, population
                                                                            FROM (
                                                                                SELECT longitude, latitude, population
                                                                                FROM (
                                                                                    SELECT longitude, latitude, population
                                                                                    FROM (
                                                                                        SELECT longitude, latitude, population
                                                                                        FROM (
                                                                                            SELECT longitude, latitude, population
                                                                                            FROM (
                                                                                                SELECT longitude, latitude, population
                                                                                                FROM (
                                                                                                    SELECT longitude, latitude, population
                                                                                                    FROM (
                                                                                                        SELECT longitude, latitude, population
                                                                                                        FROM (
                                                                                                            SELECT longitude, latitude, population
                                                                                                            FROM (
                                                                                                                SELECT longitude, latitude, population
                                                                                                                FROM (
                                                                                                                    SELECT longitude, latitude, population
                                                                                                                    FROM (
                                                                                                                        SELECT longitude, latitude, population
                                                                                                                        FROM (
                                                                                                                            SELECT longitude, latitude, population
                                                                                                                            FROM (
                                                                                                                                SELECT longitude, latitude, population
                                                                                                                                FROM (
                                                                                                                                    SELECT longitude, latitude, population
                                                                                                                                    FROM (
                                                                                                                                        SELECT longitude, latitude, population
                                                                                                                                        FROM (
                                                                                                                                            SELECT longitude, latitude, population
                                                                                                                                            FROM (
                                                                                                                                                SELECT longitude, latitude, population
                                                                                                                                                FROM (
                                                                                                                                                    SELECT longitude, latitude, population
                                                                                                                                                    FROM (
                                                                                                                                                        SELECT longitude, latitude, population
                                                                                                                                                        FROM (
                                                                                                                                                            SELECT longitude, latitude, population