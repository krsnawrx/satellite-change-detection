from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj='../satellite-change-detection/models/best_model.pth',
    path_in_repo='best_model.pth',
    repo_id='krsnawrx/bihar-flood-mapper',
    repo_type='model'
)
print('Model uploaded successfully')