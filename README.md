# Text Classification: Abusive Language

An academic project exploring the use of convolutional neural networks and variants of word embeddings to detect abusive language in online text.

## Current Status

The current state of this repo is only the first step in the beginning to organize the material for open-sourcing. The thinking is to just get it committed, then organize and refine as the project wraps up.

## TODO

- Add doc comments to top of files.
- Update GC-Commands-Reference.txt to properly represent commands used in the project.
- Flesh out a README for each of the directories in the project.
- Move the notebooks into the appropriate directories.
- Clean up the tfidf directory in order to include it in the repo.

## Changelog
All notable changes to this project will be documented here.

### 2019-12-12
#### Added:
- GoogleNews-related files and directory to data directory.
- Jigsaw-related files and directory to data directory.  
- PNGs of diagrams used in paper.
- Lexicon-related files and directory to the data directory.
- Specialized embeddings-related files and directories.
- Wikipedia Personal Attacks corpus-related files and directory.
#### Changed:
- Moved old Gao feature extraction script to deprecated directory.
- Moved experiment scripts to experiments directory. 
- moved runner scripts to runners directory in experiments directory.

### 2019-12-06
#### Added:
- Docs directory.
- Diagrams in the docs directory. 
#### Changed: 
- Renamed runner scripts to example_SCRIPT_NAME.sh.
- Abstracted runner scripts to be examples.
- Fixed variable references in runner scripts.
- Removed dependence on logger from cnn_text.py.
- Moved old versions of networks and experiments to deprecated directory.

### 2019-11-25
#### Added:
- Experiment scripts.
- Deprecated scripts.
- Preprocessing scripts.
- Local runner shell script.
- Google Cloud Service runner script.
- Gitignore file.
- README file in each directory.
- Jupyter notebooks with related work.
