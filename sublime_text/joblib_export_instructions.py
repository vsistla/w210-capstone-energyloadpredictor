#exporting the departure encoder
output = open('xgb_label_encoder.pkl', 'wb')
pickle.dump(lbl, output)
output.close()

# importing encoder
aws_access_key_id = 'AKIATAVK2UELBEVSLANM'
aws_secret_access_key = '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'

s3_client = boto3.client('s3', 
                         aws_access_key_id=aws_access_key_id, 
                         aws_secret_access_key=aws_secret_access_key)

bucket_name = 'ipowermigrid.monthly.models'
key = 'xgb_label_encoder.pkl'
# read model from S3 bucket
with tempfile.TemporaryFile() as fp:
    s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
    fp.seek(0)
    lbl = joblib.load(fp)