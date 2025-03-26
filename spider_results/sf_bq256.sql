SELECT SUM(balance) FROM (
    SELECT 
        CASE 
            WHEN transaction_type = 'delegatecall' OR transaction_type = 'callcode' OR transaction_type = 'staticcall' THEN 0
            ELSE balance
        END AS balance
    FROM (
        SELECT 
            transaction_id,
            transaction_type,
            transaction_timestamp,
            transaction_from_address,
            transaction_to_address,
            transaction_value,
            transaction_gas_used,
            transaction_gas_price,
            transaction_fee,
            transaction_block_number,
            transaction_block_hash,
            transaction_receipt_status,
            transaction_receipt_cumulative_gas_used,
            transaction_receipt_contract_address,
            transaction_receipt_root,
            transaction_receipt_logs_bloom,
            transaction_receipt_logs,
            transaction_receipt_transaction_index,
            transaction_receipt_transaction_hash,
            transaction_receipt_contract_address,
            transaction_receipt_root,
            transaction_receipt_logs_bloom,
            transaction_receipt_logs,
            transaction_receipt_transaction_index,
            transaction_receipt_transaction_hash,
            transaction_receipt_contract_address,
            transaction_receipt_root,
            transaction_receipt_logs_bloom,
            transaction_receipt_logs,
            transaction_receipt_transaction_index,
            transaction_receipt_transaction_hash,
            transaction_receipt_contract_address,
            transaction_receipt_root,
            transaction_receipt_logs_bloom,
            transaction_receipt_logs,
            transaction_receipt_transaction_index,
            transaction_receipt_transaction_hash,
            transaction_receipt_contract_address,
            transaction_receipt_root,
            transaction_receipt_logs_bloom,
            transaction_receipt_logs,
            transaction_receipt_transaction_index,
            transaction_receipt_transaction_hash,
            transaction_receipt_contract_address,
            transaction_receipt_root,
            transaction_receipt_logs_bloom,
            transaction_receipt_logs,
            transaction_receipt_transaction_index,
            transaction_receipt_transaction_hash,
            transaction_receipt_contract_address,
            transaction_receipt_root,
            transaction_receipt_logs_bloom,
            transaction_receipt_logs,
            transaction_receipt_transaction_index,
            transaction_receipt_transaction_hash,
            transaction_receipt_contract_address,
            transaction_receipt_root,
            transaction_receipt_logs_bloom,
            transaction_receipt_logs,
            transaction_receipt_transaction_index,
            transaction_receipt_transaction_hash,
            transaction_receipt_contract_address,
            transaction_receipt_root,
            transaction_receipt_logs_bloom,
            transaction_receipt_logs,
            transaction_receipt_transaction_index,
            transaction_receipt_transaction_hash,
            transaction_receipt_contract_address,
            transaction_receipt_root,
            transaction_receipt_logs_bloom,
            transaction_receipt_logs,
            transaction_receipt_transaction_index,
            transaction_receipt_transaction_hash,
            transaction_receipt_contract_address,
            transaction_receipt_root,
            transaction_receipt_logs_bloom,
            transaction_receipt_logs,
            transaction_receipt_transaction_index,
            transaction_receipt_transaction_hash,
            transaction_receipt_contract_address,
            transaction