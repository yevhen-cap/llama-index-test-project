#! /bin/sh

if [[ ! -d ".venv" ]]
then
    echo "\033[31mVirtual environment not found\033[0m"
    echo "\033[32mCreating Virtual environment\033[0m"
    python3 -m venv .venv
    echo "\033[32Activating Virtual environment\033[0m"
    source .venv/bin/activate
    echo "\033[32mInstalling dependencies\033[0m"
    pip3 install .
fi

if [[ ! $VIRTUAL_ENV ]]
then
    echo "\033[32mActivating Virtual environment\033[0m"
    source .venv/bin/activate
fi

if [[ ! -f "resume-dataset.zip" ]]
then
    echo "\033[32mDownloading data set\033[0m"
    curl -L -o resume-dataset.zip\
      https://www.kaggle.com/api/v1/datasets/download/snehaanbhawal/resume-dataset
    
    echo "\033[32mExtracting test data\033[0m"
    unzip -o resume-dataset.zip 
fi

echo "\033[32mSetting env variables\033[0m"
export $(cat .env | xargs)

if [[ ! -d "db/${INDEX_NAME}" ]]
then
    echo "\033[32mIngest data into ChromaDB\033[0m"
    python3 ingest_chroma.py
fi

echo "\033[32mStarting application\033[0m"
streamlit run main.py 
