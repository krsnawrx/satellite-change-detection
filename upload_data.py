from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj='../satellite-change-detection/data/patna_before_flood_2023.tif',
    path_in_repo='patna_before_flood_2023.tif',
    repo_id='krsnawrx/bihar-flood-mapper',
    repo_type='model'
)

api.upload_file(
    path_or_fileobj='../satellite-change-detection/data/patna_after_flood_2023.tif',
    path_in_repo='patna_after_flood_2023.tif',
    repo_id='krsnawrx/bihar-flood-mapper',
    repo_type='model'
)

print('Done')