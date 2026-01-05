
# Feature Engineering: Geographic Risk Signals
# Based on False Negative analysis - discovered hardcoded S1 rules

def add_geographic_risk_features(df):
    """
    Add geographic risk features based on S1 rule reverse-engineering.
    These features capture combination rules that S1 uses but ML model doesn't have.

    Features added:
    - is_high_risk_location: City+subdistrict appears only in REJECTED (100% fraud rate)
    - is_bogor: BOGOR city has 7x fraud overrepresentation
    - is_bogor_single: BOGOR + Single marital status (high FN concentration)
    - is_high_risk_city: Cities with elevated fraud rates
    - is_clean_profile_pattern: Clean fraud profile (zero duplicates + BOGOR + Single)

    Usage:
        df = add_geographic_risk_features(df)
    """
    import pandas as pd

    # High-risk city-subdistrict combinations (100% fraud rate in historical data)
    HIGH_RISK_LOCATIONS = {
        # Add your discovered combinations here
        # Example: 'BOGOR + CILEUNGSI', 'BOGOR + BOJONG_GEDE', etc.
    }

    # High-risk cities (elevated fraud rates)
    HIGH_RISK_CITIES = ['BOGOR', 'LAMPUNG_TENGAH']

    df_copy = df.copy()

    # Feature 1: High-risk location (city + subdistrict)
    if 'payload_address_subdistrict' in df_copy.columns:
        df_copy['city_subdistrict_combo'] = (
            df_copy['payload_address_city'].astype(str) + ' + ' + 
            df_copy['payload_address_subdistrict'].astype(str)
        )
        df_copy['is_high_risk_location'] = (
            df_copy['city_subdistrict_combo'].isin(HIGH_RISK_LOCATIONS).astype(int)
        )
    else:
        df_copy['is_high_risk_location'] = 0

    # Feature 2: BOGOR city indicator
    df_copy['is_bogor'] = (df_copy['payload_address_city'] == 'BOGOR').astype(int)

    # Feature 3: BOGOR + Single combination
    df_copy['is_bogor_single'] = (
        (df_copy['payload_address_city'] == 'BOGOR') & 
        (df_copy['payload_ktp_maritalstatus'] == 'S')
    ).astype(int)

    # Feature 4: High-risk city indicator
    df_copy['is_high_risk_city'] = (
        df_copy['payload_address_city'].isin(HIGH_RISK_CITIES).astype(int)
    )

    # Feature 5: Clean profile fraud pattern
    if 'spouse_id_cnt_ktp' in df_copy.columns:
        df_copy['is_clean_profile_pattern'] = (
            (df_copy['spouse_id_cnt_ktp'] == 0) & 
            (df_copy['payload_address_city'] == 'BOGOR') & 
            (df_copy['payload_ktp_maritalstatus'] == 'S')
        ).astype(int)
    else:
        df_copy['is_clean_profile_pattern'] = 0

    return df_copy


# Integration into training pipeline:
# 1. Apply to training data BEFORE model training
# 2. These features become part of the model input
# 3. Model will learn when to rely on these signals

# Example usage in training:
if __name__ == '__main__':
    import pandas as pd

    # Load training data
    train_df = pd.read_csv('training_data.csv')

    # Add geographic risk features
    train_df = add_geographic_risk_features(train_df)

    # Continue with model training...
    # X = train_df[feature_columns + ['is_high_risk_location', 'is_bogor_single', ...]]
    # model.fit(X, y)
