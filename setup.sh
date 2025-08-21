apt-get update
apt-get install -y ffmpeg
apt-get install -y espeak-ng
pip install --upgrade pip
pip install git+https://github.com/descriptinc/audiotools
pip install huggingface_hub vocos datasets
pip install -e .
pip install -U numpy

git config --global user.email zxcv05999@naver.com
git config --global user.name thxxx

wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvjf LJSpeech-1.1.tar.bz2

python3 setdata.py

wget https://huggingface.co/Daniel777/personals/resolve/main/matcha_ljspeech.ckpt?download=true
mv matcha_ljspeech.ckpt?download=true /workspace/matcha_ljspeech.ckpt