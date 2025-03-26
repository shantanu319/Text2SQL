SELECT SUM(balance / POWER(10, 18)) AS total_circulating_supply FROM (
    SELECT sender_address AS address, SUM(amount_received - amount_sent) AS balance
    FROM transactions
    WHERE sender_address != '0x000...' AND receiver_address != '0x000...'
    GROUP BY sender_address