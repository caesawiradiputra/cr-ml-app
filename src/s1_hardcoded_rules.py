
# S1 Hardcoded Rules - Discovered from False Negative Analysis
# Generated: 2026-01-05 14:40:47
# These rules explain 19/28 False Negative cases

def check_s1_hardcoded_rules(row):
    """
    Check if a transaction matches any discovered S1 hardcoded rules.

    Args:
        row: pandas Series with payload_* features

    Returns:
        dict with 'is_flagged' (bool) and 'matched_rules' (list)
    """
    matched_rules = []

    # Rule 1: 28 + CIBEDUG
    if row['payload_application_soa_id'] == '28' and row['payload_ktp_subdistrict'] == 'CIBEDUG':
        matched_rules.append({
            'rule_id': 1,
            'pattern': '28 + CIBEDUG',
            'confidence': 'HIGH'
        })

    # Rule 2: BOGOR + SUKASARI + S
    if row['payload_address_city'] == 'BOGOR' and row['payload_ktp_subdistrict'] == 'SUKASARI' and row['payload_ktp_maritalstatus'] == 'S':
        matched_rules.append({
            'rule_id': 2,
            'pattern': 'BOGOR + SUKASARI + S',
            'confidence': 'HIGH'
        })

    # Rule 3: BOGOR + CITAPEN
    if row['payload_address_city'] == 'BOGOR' and row['payload_ktp_subdistrict'] == 'CITAPEN':
        matched_rules.append({
            'rule_id': 3,
            'pattern': 'BOGOR + CITAPEN',
            'confidence': 'HIGH'
        })

    # Rule 4: 2307NC0005 + CILEUNGSI
    if row['payload_agent_marketingid'] == '2307NC0005' and row['payload_ktp_subdistrict'] == 'CILEUNGSI':
        matched_rules.append({
            'rule_id': 4,
            'pattern': '2307NC0005 + CILEUNGSI',
            'confidence': 'HIGH'
        })

    # Rule 5: 2405NC0005 + CILEUNGSI
    if row['payload_agent_marketingid'] == '2405NC0005' and row['payload_ktp_subdistrict'] == 'CILEUNGSI':
        matched_rules.append({
            'rule_id': 5,
            'pattern': '2405NC0005 + CILEUNGSI',
            'confidence': 'HIGH'
        })

    # Rule 6: 2403NC0005 + CIAWI
    if row['payload_agent_marketingid'] == '2403NC0005' and row['payload_ktp_subdistrict'] == 'CIAWI':
        matched_rules.append({
            'rule_id': 6,
            'pattern': '2403NC0005 + CIAWI',
            'confidence': 'HIGH'
        })

    # Rule 7: 2408NC0002 + CIAWI
    if row['payload_agent_marketingid'] == '2408NC0002' and row['payload_ktp_subdistrict'] == 'CIAWI':
        matched_rules.append({
            'rule_id': 7,
            'pattern': '2408NC0002 + CIAWI',
            'confidence': 'HIGH'
        })

    # Rule 8: 2408NC0002 + PADASUKA
    if row['payload_agent_marketingid'] == '2408NC0002' and row['payload_ktp_subdistrict'] == 'PADASUKA':
        matched_rules.append({
            'rule_id': 8,
            'pattern': '2408NC0002 + PADASUKA',
            'confidence': 'HIGH'
        })

    # Rule 9: 2405NC0005 + PAGELARAN
    if row['payload_agent_marketingid'] == '2405NC0005' and row['payload_ktp_subdistrict'] == 'PAGELARAN':
        matched_rules.append({
            'rule_id': 9,
            'pattern': '2405NC0005 + PAGELARAN',
            'confidence': 'HIGH'
        })

    # Rule 10: KERTAWINANGUN + S
    if row['payload_ktp_subdistrict'] == 'KERTAWINANGUN' and row['payload_ktp_maritalstatus'] == 'S':
        matched_rules.append({
            'rule_id': 10,
            'pattern': 'KERTAWINANGUN + S',
            'confidence': 'HIGH'
        })

    # Rule 11: BOGOR + 28 + SIRNAGALIH
    if row['payload_address_city'] == 'BOGOR' and row['payload_application_soa_id'] == '28' and row['payload_ktp_subdistrict'] == 'SIRNAGALIH':
        matched_rules.append({
            'rule_id': 11,
            'pattern': 'BOGOR + 28 + SIRNAGALIH',
            'confidence': 'HIGH'
        })

    # Rule 12: BOGOR + 2403NC0005 + PADASUKA
    if row['payload_address_city'] == 'BOGOR' and row['payload_agent_marketingid'] == '2403NC0005' and row['payload_ktp_subdistrict'] == 'PADASUKA':
        matched_rules.append({
            'rule_id': 12,
            'pattern': 'BOGOR + 2403NC0005 + PADASUKA',
            'confidence': 'HIGH'
        })

    # Rule 13: BOGOR + PARAKAN + M
    if row['payload_address_city'] == 'BOGOR' and row['payload_ktp_subdistrict'] == 'PARAKAN' and row['payload_ktp_maritalstatus'] == 'M':
        matched_rules.append({
            'rule_id': 13,
            'pattern': 'BOGOR + PARAKAN + M',
            'confidence': 'HIGH'
        })

    # Rule 14: 2403NC0005 + PAGELARAN + M
    if row['payload_agent_marketingid'] == '2403NC0005' and row['payload_ktp_subdistrict'] == 'PAGELARAN' and row['payload_ktp_maritalstatus'] == 'M':
        matched_rules.append({
            'rule_id': 14,
            'pattern': '2403NC0005 + PAGELARAN + M',
            'confidence': 'HIGH'
        })

    return {
        'is_flagged': len(matched_rules) > 0,
        'matched_rules': matched_rules,
        'rule_count': len(matched_rules)
    }

# Usage example:
# result = check_s1_hardcoded_rules(transaction_row)
# if result['is_flagged']:
#     print(f"Flagged by {result['rule_count']} rule(s)")
