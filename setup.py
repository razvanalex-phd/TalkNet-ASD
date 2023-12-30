from setuptools import setup

setup(
    name="TalkNetASD",
    version="1.0",
    description="Packaged version of TaskNet-ASD",
    long_description=open("README.md").read(),
    packages=[
        "TalkNetASD",
        "TalkNetASD.model",
        "TalkNetASD.model.faceDetector",
        "TalkNetASD.model.faceDetector.s3fd",
        "TalkNetASD.TalkSet",
        "TalkNetASD.utils",
    ],
    install_requires=[
        "torch>=1.6.0",
        "torchaudio>=0.6.0",
        "numpy",
        "scipy",
        "scikit-learn",
        "tqdm",
        "scenedetect",
        "opencv-python",
        "python_speech_features",
        "torchvision",
        "ffmpeg",
        "gdown",
        "youtube-dl",
    ],
)
