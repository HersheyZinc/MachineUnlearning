import glob
from tqdm import tqdm

def clean_text():
    for file_path in tqdm(glob.glob("./data/HarryPotter/*.txt")):
        with open(file_path, "r", encoding="UTF-8") as f:
            text = f.read()

        text = text.replace("\n\n", "\n")
        text = text.replace("\n\n", "\n")
        text = text.replace("\n\n", "\n")

        with open(file_path.replace("/HarryPotter", "/HarryPotterClean"), "w", encoding="UTF-8") as f:
            f.write(text)

def load_dataset():
    dataset = load_dataset("text", data_files=glob.glob("./data/HarryPotterClean/*.txt"), sample_by="line")
    return dataset