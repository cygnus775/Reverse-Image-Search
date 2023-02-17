
# Reverse Image Search

A reverse image search application that extracts image features and uses Approximate Nearest Neighbours to find similar images.

## Run Locally

To get started, clone the repo

```bash
  git clone https://github.com/cygnus775/Reverse-Image-Search.git
```
Install required packages
```bash
pip install -r requirements.txt
```
Create the following directories
```bash
mkdir -p data/raw data/processed
```

Place the images to be searched in the `data/raw` folder. Once the images are placed correctly, extract the image features by running

```bash
python generate.py
```

Once the feature extraction completes, the fastapi server can be run by
```bash
uvicorn main:app
```
To run the frontend for demo purposes, run 
```bash
streamlit run streamlit_app/webui.py
```
Configurations can be found at `configs/config.json`


## Demo

![Streamlit Demo](https://raw.githubusercontent.com/cygnus775/Reverse-Image-Search/main/screenshots/demo.gif)


## License

[GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/)

