from pyspark.ml.feature import VectorAssembler, VarianceThresholdSelector, UnivariateFeatureSelector, StandardScaler, MinMaxScaler


def vectorize(dataframe):

    input_columns = [
        # "Flow ID",
        # "Src IP",
        #"Src Port",
        # "Dst IP",
        #"Dst Port",
        #"Protocol",
        # "Timestamp",
        "Flow Duration",
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
        "Fwd Pkt Len Max",
        "Fwd Pkt Len Min",
        "Fwd Pkt Len Mean",
        "Fwd Pkt Len Std",
        "Bwd Pkt Len Max",
        "Bwd Pkt Len Min",
        "Bwd Pkt Len Mean",
        "Bwd Pkt Len Std",
        "Flow Byts/s",
        "Flow Pkts/s",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Tot",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Tot",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Fwd Header Len",
        "Bwd Header Len",
        "Fwd Pkts/s",
        "Bwd Pkts/s",
        "Pkt Len Min",
        "Pkt Len Max",
        "Pkt Len Mean",
        "Pkt Len Std",
        "Pkt Len Var",
        "SYN Flag Cnt",
        "RST Flag Cnt",
        "PSH Flag Cnt",
        "ACK Flag Cnt",
        "CWE Flag Count",
        "ECE Flag Cnt",
        "Down/Up Ratio",
        "Pkt Size Avg",
        "Fwd Seg Size Avg",
        "Bwd Seg Size Avg",
        "Subflow Fwd Pkts",
        "Subflow Fwd Byts",
        "Subflow Bwd Pkts",
        "Subflow Bwd Byts",
        "Init Fwd Win Byts",
        "Init Bwd Win Byts",
        "Fwd Act Data Pkts",
        "Fwd Seg Size Min",
        "Active Mean",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min",
    ]

    assembler = VectorAssembler(inputCols=input_columns, outputCol="features")
    return assembler.transform(dataframe)


def standard_feature_select(dataframe, select_top=30):

    # apply z-score normalization
    dataframe = StandardScaler(inputCol="features", outputCol="new_features", withStd=True, withMean=False) \
        .fit(dataframe).transform(dataframe).drop("features").withColumnRenamed("new_features", "features")

    # select top 30 based on hypothesis testing (ANOVA)
    selector = UnivariateFeatureSelector(outputCol="new_features", labelCol='label', selectionMode="numTopFeatures")
    selector.setFeatureType("continuous").setLabelType("categorical").setSelectionThreshold(select_top)
    dataframe = selector.fit(dataframe).transform(dataframe).drop("features").withColumnRenamed("new_features", "features")

    return dataframe

def minmax_feature_select(dataframe, select_top=30):

    # apply min-max standarization
    dataframe = MinMaxScaler(inputCol="features", outputCol="new_features") \
        .fit(dataframe).transform(dataframe).drop("features").withColumnRenamed("new_features", "features")

    return dataframe
