import os
import requests
import tarfile
import shutil
import numpy as np
from skimage import io
from multiprocessing import Pool
import warnings
import logging

def get_files():
    
    # URL of the file to be downloaded
    imgs_url = 'https://zenodo.org/records/1476551/files/trainingsetv1d1.tar.gz?download=1'
    csv_url = "https://zenodo.org/records/1476551/files/trainingset_v1d1_metadata.csv?download=1"
    
    # Call the download function to download the url and name the file
    download_files(imgs_url, 'trainingsetv1d1.tar.gz')
    download_files(csv_url, 'trainingset_v1d1_metadata.csv')

def download_files(url, filename):

    try:
        # Downloads the url
        with requests.get(url, stream=True) as response:
            response.raise_for_status()

            # Download the file in chunks for better downloads
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            total_downloaded = 0
            chunk_size = 8192  # 8 KB per chunk

            # Write the download to a file
            with open('../data/' + filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        total_downloaded += len(chunk)
                        downloaded_gb = total_downloaded / (1024 * 1024 * 1024)  # Convert to GB
                        print(f"\r{filename}: {downloaded_gb:.2f} GB downloaded...", end="")
            print('')

    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"OOps: Something Else: {err}")

def check_files():

    # Check if any files exist in the directory
    directory_path = '../data/'
    files_in_directory = os.listdir(directory_path)

    if files_in_directory:
        return True
    else:
        return False

def extract_tar_gz():
 
    # Extracts the tar.gz file that was downloaded
    print('Extracting trainingsetv1d1.tar.gz')
    extract_path = ('../data/')
    file_path = ('../data/trainingsetv1d1.tar.gz')

    # Ensure the extract path exists
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    # Open the .tar.gz file
    with tarfile.open(file_path, 'r:gz') as tar:
        # Extract all the contents into the directory
        tar.extractall(path=extract_path)
        print(f"Extracted all contents of {file_path} to {extract_path}")

def split_data(train_size=0.8, val_size=0.1, test_size=0.1):

    source_folder = '../data/TrainingSet'

    # Find all the classes in the training set
    classes = os.listdir(source_folder)
    for cls in classes:
        cls_folder = os.path.join(source_folder, cls)
        if not os.path.isdir(cls_folder):

            continue
        
        # Shuffle the training set data
        files = os.listdir(cls_folder)
        np.random.seed(220301)
        np.random.shuffle(files)

        # Create the split indices
        train_end = int(len(files) * train_size)
        val_end = train_end + int(len(files) * val_size)

        # Define start index for each dataset
        dataset_indices = {'train': 0, 'validation': train_end, 'test': val_end}

        # Create new files for the split data and send data to file locations
        for set_type, start_idx in dataset_indices.items():
            end_idx = len(files) if set_type == 'test' else start_idx + int(len(files) * (val_size if set_type == 'validation' else train_size))
            dest_folder = os.path.join(source_folder, '..', set_type, cls)
            os.makedirs(dest_folder, exist_ok=True)

            for file in files[start_idx:end_idx]:
                shutil.move(os.path.join(cls_folder, file), os.path.join(dest_folder, file))

def clear_folder(path):

    # Check if the path exists
    if os.path.exists(path):
        if os.path.isfile(path) or os.path.islink(path):
            # Path is a file or a link, remove it
            try:
                os.unlink(path)
                print(f"File {path} has been removed successfully.")
            except Exception as e:
                print(f'Failed to delete {path}. Reason: {e}')
        elif os.path.isdir(path):
            # Path is a directory, remove its contents and the directory
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

            # Remove the folder itself
            try:
                os.rmdir(path)
                print(f"Folder {path} has been removed successfully.")
            except OSError as e:
                print(f"Error: {path} : {e.strerror}")
    else:
        print(f"Path {path} does not exist.")

def clear_all_folders():

    # Get rid of all folders in the directory
    data_directory = '../data/'

    if not os.path.exists(data_directory) or not os.path.isdir(data_directory):
        print(f"The directory {data_directory} does not exist.")
        return

    for item in os.listdir(data_directory):
        item_path = os.path.join(data_directory, item)
        if os.path.isdir(item_path):
            clear_folder(item_path)

def process_images_in_class(class_path, x, y):
    """
    Worker function to process a single class path
    """

    # Ignore numerous warnings 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Get file names
        for image_file in os.listdir(class_path):

            # Ignore hidden files
            if image_file.startswith('._'):
                os.remove(os.path.join(class_path, image_file))
                continue

            # Join class path and image path
            image_path = os.path.join(class_path, image_file)

            # Standardise for file reading 
            if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Change shape of the image according to the dataset creators
                    image_data = io.imread(image_path)
                    image_data = image_data[x[0]:x[1], y[0]:y[1], :3]
                    io.imsave(image_path, image_data)
                    print(f"\rProcessing: {image_path}", end="")
                except Exception as e:
                    os.remove(os.path.join(class_folder, image_file))
                    continue

def process_images_in_classes(num_workers=4):
    '''
    Multiprocessing to process multiple classes at the same time!
    '''
    directory = '../data/TrainingSet'
    x = [66, 532]
    y = [105, 671]
    
    # Creates a list of Folders with files paths 
    class_paths = [os.path.join(directory, class_dir) for class_dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, class_dir))]

    # Create a pool of workers and distribute the processing of each class
    with Pool(num_workers) as pool:

        # Map the worker function to each class directory
        pool.starmap(process_images_in_class, [(class_path, x, y) for class_path in class_paths])

def download_extract_split():
    
    clear_folder('../data/trainingset_v1d1_metadata.csv')
    clear_all_folders()
    get_files()
    extract_tar_gz()
    process_images_in_classes((os.cpu_count() -1))
    split_data()
    clear_folder('../data/TrainingSet')
    clear_folder('../data/trainingsetv1d1.tar.gz')
