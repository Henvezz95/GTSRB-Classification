import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files('meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/', path='./', unzip=True, quiet=False)