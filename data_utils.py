from datetime import datetime
import io
import os.path
import json
import argparse
from typing import Union
import gdown
import re
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from concurrent.futures import ThreadPoolExecutor, as_completed

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def authenticate(token_file="token.json", credentials_file="credentials.json"):
    """Authenticate and return Google Drive credentials.
    
    Args:
        token_file: Path to the token file for storing credentials
        credentials_file: Path to the OAuth2 credentials file
        
    Returns:
        Credentials object for Google Drive API
    """
    creds = None

    # Load existing token if present
    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        except ValueError:
            # Bad or incompatible token file
            creds = None

    # If there are no (valid) credentials available, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file,
                SCOPES,
                redirect_uri="http://localhost"
            )

            # ---- MANUAL CONSOLE AUTH FLOW (works with oauthlib 1.x) ----
            auth_url, _ = flow.authorization_url(
                access_type="offline",
                prompt="consent",
            )

            print("\nPlease go to this URL and authorize the application:\n")
            print(auth_url)
            print()

            code = input("Enter the authorization code: ").strip()
            flow.fetch_token(code=code)
            creds = flow.credentials
            # ------------------------------------------------------------

        # Save credentials for next run
        with open(token_file, "w") as token:
            token.write(creds.to_json())

    return creds


def connect(creds):
    """Build and return a Google Drive service object.
    
    Args:
        creds: Credentials object from authenticate()
        
    Returns:
        Google Drive service object
    """
    try:
        service = build("drive", "v3", credentials=creds)
        return service
    except HttpError as error:
        print(f"An error occurred while connecting: {error}")
        raise

def list_folder_contents(service, folder_id):
    """List all files in a Google Drive folder.
    
    Args:
        service: Google Drive service object
        folder_id: ID of the folder to list contents from
        
    Returns:
        List of file metadata dictionaries
    """
    results = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType)",
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                pageToken=page_token,
            )
            .execute()
        )

        results.extend(response.get("files", []))
        page_token = response.get("nextPageToken")

        if not page_token:
            break

    return results


def download_npz_file(creds, file_id):
    service = connect(creds)

    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    fh.seek(0)
    return np.load(fh)


def download_npz_files_parallel(creds, file_ids: list[str],
                                layer_names: list[str],
                                max_workers: int = 4) -> list[tuple[str, np.ndarray]]:
    """
    Download a list of npz files in parallel.

    Args:
        creds: Credentials object
        file_ids: List of file IDs
        layer_names: List of layer names
        max_workers: Maximum number of workers

    Returns:
        List of tuples containing the layer name and the activations
        (layer_name, activations)
    """
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_npz_file, creds, fid): fid
            for fid in file_ids
        }

        for future in as_completed(futures):
            fid = futures[future]
            try:
                results[fid] = future.result()
            except Exception as e:
                print(f"Failed to download {fid}: {e}")

    output = list()
    for key, value in results.items():
        name = layer_names[file_ids.index(key)].replace("layer_", "").replace(".npz", "")
        output.append((name, value['data']))

    return output

def process_activations_with_padding(activations: list[np.ndarray], 
    max_seq_len: int,
    padding_side:str="left",
    truncation_side:str="left") -> list[np.ndarray]:
    """
    Process a list of activations with padding to a given sequence length.
    """

    # Scan shapes to find the max sequence in array:
    max_in_array_seq_len = max([act.shape[1] for act in activations])
    if max_in_array_seq_len <= max_seq_len:
        max_seq_len = max_in_array_seq_len
    
    # Process each activation:
    output_activations = []
    for act in activations:
        seq_len = act.shape[1]

        # Do truncation
        if seq_len > max_seq_len:
            if truncation_side == "right":
                output_activations.append(act[:max_seq_len])
            else:
                output_activations.append(act[-max_seq_len:])
            continue

        # Do padding
        pad_width = max_seq_len - seq_len
        pad_shape = np.zeros((act.shape[0], pad_width, act.shape[2]), dtype=act.dtype)

        if padding_side == "left":
            output_activations.append(np.concatenate([pad_shape, act], axis=1))
        else:
            output_activations.append(np.concatenate([act, pad_shape], axis=1))
    return np.stack(output_activations, axis=0)
    

def download_activations_dataset( 
    split: str,
    questions: list[int],
    model_name: str,
    layers: list[int],
    output_dir: str,
    root_folder_id="17o-gw45qAwmr0rQQPMANCas-oQ1KEXr5",
    token_file="token.json",
    credentials_file="credentials.json",
    max_seq_len: int = 512,
    num_workers: int = 6,
    ):
    """Download the activations dataset for a given split, questions, model, and layers.

    TODO: Include some logic to do all questions and all layers...
    
    Args:
        split: Split name
        questions: List of question IDs
        model_name: Model name
        layers: List of layer indices
        output_dir: Output directory
        root_folder_id: Root folder ID
    """

    # Authenticate and connnect:
    creds = authenticate(token_file=token_file,
     credentials_file=credentials_file)
    service = connect(creds)

    # List the folder contents: Split
    files = list_folder_contents(service, root_folder_id)

    split_file_id = None
    for i,file in enumerate(files):
        if file["name"] == split:
            split_file_id = file["id"]
            break
    if split_file_id is None:
        raise ValueError(f"Split {split} not found in the root folder, file names: {[file['name'] for file in files]}")

    # List the folder contents: Questions
    questions_files = list_folder_contents(service, split_file_id)
    question_file_ids_and_names = list()

    if len(questions) == 1 and questions[0] == -1:
        question_file_ids_and_names = [(file["id"], file["name"]) for file in questions_files]
        print(f"Found {len(question_file_ids_and_names)} questions for split:{split}")
    else:
        required_question_file_names = [f"question_{q}" for q in questions]
        for i, file in enumerate(questions_files):
            if file["name"] in required_question_file_names:
                question_file_ids_and_names.append((file["id"], file["name"]))
        print(f"Found {len(question_file_ids_and_names)} questions of {len(required_question_file_names)} for split: {split}")

    # For each question file, download the model name and layer:
    questions = list()
    all_layer_activations = list()
    all_layer_ids = list()
    for question_file_id, question_file_name in tqdm(question_file_ids_and_names,
     desc="Downloading questions"):

        model_name_files = list_folder_contents(service, question_file_id)
        
        model_name_file_id = None
        for file in model_name_files:
            if file["name"] in model_name:
                model_name_file_id = file["id"]
                break
        if model_name_file_id is None:
            raise ValueError(f"Model name {model_name} not found in the question file {question_file_id}, file names: {[file['name'] for file in model_name_files]}")

        layer_files = list_folder_contents(service, model_name_file_id)
        layer_file_ids_and_names = list()

        if len(layers) == 1 and layers[0] == -1:
            layer_file_ids_and_names = [(file["id"], file["name"]) for file in layer_files]
        else:
            required_layer_file_names = [f"layer_{l}.npz" for l in layers]
            for i, file in enumerate(layer_files):
                if file["name"] in required_layer_file_names:
                    layer_file_ids_and_names.append((file["id"], file["name"]))

            assert len(layer_file_ids_and_names) == len(required_layer_file_names),\
                f"Expected {len(required_layer_file_names)} layer files, found {len(layer_file_ids_and_names)}"

        questions.append(int(question_file_name.replace("question_", "")))
        
        layer_file_ids, layer_file_names = zip(*layer_file_ids_and_names)
        activations = download_npz_files_parallel(creds,
         file_ids=layer_file_ids,
         layer_names=layer_file_names,
         max_workers=num_workers)
        
        layer_ids, layer_activations = zip(*activations)
        all_layer_activations.extend(np.stack(layer_activations))
        all_layer_ids.extend(np.array(layer_ids))

    # We stack activations here:
    all_layer_activations = process_activations_with_padding(all_layer_activations, 
                                                            max_seq_len=max_seq_len,
                                                            padding_side="left",
                                                            truncation_side="left")
    all_layer_ids = np.stack(all_layer_ids, axis=0)

    model_split_output_dir_name = f"{model_name}_{split}"
    output_dir = os.path.join(output_dir, model_split_output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "data")
    output_data = {
        'questions': questions,
        'activations': all_layer_activations,
        'layer_ids': all_layer_ids,
    }

    np.savez_compressed(output_path, **output_data)
    print(f"Saved activations to {output_path}")
    
    return True


def parse_str_list_arg(questions_arg: str) -> list[int]:
    """
    Parse a string of the form '[i,j]' and return a list of integers from i to j inclusive.
    E.g. '[0,5]' -> [0, 1, 2, 3, 4, 5]
          '[0]' -> [0]
          'all' -> [-1]
          '-1' -> [-1]
    """
    match_multi_list = re.match(r"\[\s*(\d+)\s*,\s*(\d+)\s*]", questions_arg)
    match_single_list = re.match(r"\[\s*(\d+)\s*]", questions_arg)
    match_single_number = re.match(r"^\s*(\d+)\s*$", questions_arg)


    if match_multi_list:
        start, end = map(int, match_multi_list.groups())
    elif match_single_list:
        start = end = int(match_single_list.group(1))
    elif match_single_number:
        start = end = int(match_single_number.group(1))
    elif questions_arg == "all":
        return [-1]
    elif questions_arg == "-1":
        return [-1]
    else:
        raise ValueError(f"Invalid questions argument format: {questions_arg}")

    if start > end:
        raise ValueError(f"Start ({start}) cannot be greater than end ({end}) in questions range")
    return list(range(start, end + 1))

def download_parquet_files(files: list[dict], output_dir: str):
    """
    Download all files from the list of files.
    
    Args:
        files: List of file metadata dictionaries
        output_dir: Output directory for downloaded data

    Returns:
        None
    """

    for file in tqdm(files, desc="Downloading parquet files"):
        try:
            file_id = file["id"]
            url = f"https://drive.google.com/uc?id={file_id}"
            output_path = os.path.join(output_dir, file["name"])
            gdown.download(url, output_path, quiet=False)
        except Exception as e:
            print(f"Error downloading file {file['name']}: {e}")


class ActivationsDataset(Dataset):

    def __init__(self, data_dir: str, 
                    split: str,
                    questions: list[int],
                    model_name: str,
                    layers: list[int],
                    root_folder_id: str,
                    token_file: str="token.json",
                    credentials_file: str="credentials.json",
                    max_seq_len: int=512,
                    num_download_workers: int=6,
                    redownload: bool=False):
        """
        Downloads activations dataset from Google Drive, or loads from local file if already downloaded.

        Args:
            data_dir: Directory to save downloaded files
            split: Dataset split (e.g., train, test)
            questions: Question numbers to download, e.g. '[1,2,3]' or 'all'
            model_name: Model name to download
            layers: Layer numbers to download, e.g. '[1,2,3]' or 'all'
            root_folder_id: Root folder ID
            token_file: Path to Google OAuth token.json
            credentials_file: Path to Google OAuth credentials.json
            max_seq_len: Maximum sequence length
            num_download_workers: Number of download workers
        """

        self.data_dir = data_dir
        self.split = split
        self.model_name = model_name
        self.layers = layers
        self.questions = questions
        self.root_folder_id = root_folder_id
        self.token_file = token_file
        self.credentials_file = credentials_file
        self.max_seq_len = max_seq_len
        self.num_download_workers = num_download_workers
        self.redownload = redownload

        # If the data directory for the model doesn't exist, download the data:
        data_path = os.path.join(self.data_dir, f"{self.model_name}_{self.split}", "data.npz")
        if not os.path.exists(data_path) or self.redownload:
            self.download()

        self.data = np.load(data_path)
        self.activations = self.data['activations']
        self.layer_ids = self.data['layer_ids']
        self.questions = self.data['questions']
        self.question_indices = np.arange(len(self.questions))

    def download(self):
        """Download the activations dataset from Google Drive."""

        download_activations_dataset(
            split=self.split,
            questions=self.questions,
            model_name=self.model_name,
            layers=self.layers,
            output_dir=self.data_dir,
            root_folder_id=self.root_folder_id,
            token_file=self.token_file,
            credentials_file=self.credentials_file,
            max_seq_len=self.max_seq_len,
            num_workers=self.num_download_workers,
            )

    def __getitem__(self, index):
        
        question_filter = self.questions == index
        layer_ids = self.layer_ids[question_filter]
        layer_filter = np.isin(layer_ids, self.layers)
        activations = self.activations[question_filter, layer_filter, :]

        return activations
           
class TextDataset(Dataset):

    def __init__(self,
    data_dir: str, 
    dataset_name: str,
    split: str,
    root_folder_id: str,
    token_file: str="token.json",
    credentials_file: str="credentials.json",
    redownload: bool=False):

        """
        Download text datasets from Google Drive, or load from local file if already downloaded.
        
        Args:
            data_dir: Directory to save downloaded files
            split: Dataset split (e.g., train, test)
            root_folder_id: Root folder ID
            token_file: Path to Google OAuth token.json
            credentials_file: Path to Google OAuth credentials.json
            redownload: Whether to redownload the data
        
        """
    
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split = split
        self.root_folder_id = root_folder_id
        self.token_file = token_file
        self.credentials_file = credentials_file
        self.redownload = redownload

        data_path = os.path.join(self.data_dir, f"{self.dataset_name}.parquet")

        if not os.path.exists(data_path) or self.redownload:
            self.download()

        self.data = pd.read_parquet(data_path)

        # If the data directory for the split doesn't exist, download the data:
        data_path = os.path.join(self.data_dir, f"{self.split}", "data.parquet")


    def download(self):
        """
        Download the text datasets from Google Drive.
        """

        # Authenticate and connnect:
        creds = authenticate(token_file=self.token_file,
         credentials_file=self.credentials_file)
        service = connect(creds)

        # List folder contents
        print(f"Listing contents of folder {self.root_folder_id}...")
        files = list_folder_contents(service, self.root_folder_id)

        file_names = [file["name"] for file in files]
        file_ids = [file["id"] for file in files]

        if self.dataset_name not in file_names:
            raise ValueError(f"Failed to download dataset: {self.dataset_name}, name not found in file names: {file_names}, check the root_folder_id: {self.root_folder_id}")
        else:
            file_id = file_ids[file_names.index(self.dataset_name)]

        download_parquet_files(files=[file_id], output_dir=self.data_dir)
        
    def __getitem__(self, index):
        return self.data.iloc[index]
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download THOM files from Google Drive")
    parser.add_argument("--split", type=str, default='train', help="Dataset split (e.g., train, test)")
    parser.add_argument("--model_name", type=str, default='Qwen_Qwen2-1.5B-Instruct', help="Model name to download")
    parser.add_argument("--layers", type=str, default='[0]', help="Layer numbers to download, e.g. '[1,2,3]' or 'all'")
    parser.add_argument("--questions", type=str, default='[0]', help="Question numbers to download, e.g. '[1,2,3]' or 'all'")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save downloaded files")
    parser.add_argument("--token_file", type=str, default="token.json", help="Path to Google OAuth token.json")
    parser.add_argument("--credentials_file", type=str, default="credentials.json", help="Path to Google OAuth credentials.json")
    args = parser.parse_args()
    
    main(args)
        