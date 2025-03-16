# Movie Recommendation System ML

# Libs
- Polars
  - read csv tables 
- TfidfVectorizer
  - convert a collection of text documents into a matrix of TF-IDF values (Term Frequency-Inverse Document Frequency) 
- cosine_similarity
  - measures the similarity between two vectors based on the cosine of the angle between them. This metric is widely used to compare documents represented by TF-IDF vectors
 
# How to run:
1. clone repository
2. DownLoad the files [here](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv) rating.csv and keywords.csv, after that place in the root of the project
3. install libs
   -  ` pip install polars scikit-learn numpy pandas`
4. run
   - `python3 script.py`
 

<img width="682" alt="Screenshot 2025-03-16 at 18 46 09" src="https://github.com/user-attachments/assets/358447fe-5f87-4ca0-a9b9-c19700941927" />
