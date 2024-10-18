import pandas as pd
from sentence_transformers import SentenceTransformer, util

#Principios Solid Aplicados (SRP , DIP)
 # ****SRP Cada clase tiene una responsabilidad clara ****
 # ****DIP La clase MovieSearch depende de las abstracciones EmbeddingGenerator y 
 # SimilarityCalculator, en lugar de tener las implementaciones concretas de estos métodos 
 # dentro de la clase ****

 # *** El patrón Factory es útil cuando tienes que crear objetos sin especificar la clase exacta del objeto 
 # que se va a crear. En este caso, podemos implementarlo para generar 
 # las instancias de EmbeddingGenerator, SimilarityCalculator y otras posibles clases 
 # que puedan aparecer en el futuro.***

import pandas as pd
from sentence_transformers import SentenceTransformer, util

#EDA - Analisis Exploratorio de Datos

# Clases existentes
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_data(self):
        return pd.read_csv(self.file_path)

class EmbeddingGenerator:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts, batch_size=64):
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)

class SimilarityCalculator:
    def compute_similarity(self, embedding1, embedding2):
        return util.cos_sim(embedding1, embedding2).item()

class MovieSearch:
    def __init__(self, df, embedding_generator, similarity_calculator):
        self.df = df
        self.embedding_generator = embedding_generator
        self.similarity_calculator = similarity_calculator

    def search_by_multiple_fields(self, query, fields=['Title', 'Genre']):
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        self.df['combined_text'] = self.df[fields].apply(lambda x: ' '.join(x.dropna()), axis=1)
        self.df['similarity'] = self.df.apply(
            lambda row: self.similarity_calculator.compute_similarity(row['embeddings'], query_embedding), axis=1
        )
        return self.df.sort_values(by='similarity', ascending=False)

# Implementación del patrón Factory
class MovieComponentFactory:
    """Factory para crear componentes del sistema de búsqueda de películas"""
    @staticmethod
    def create_data_loader(file_path):
        return DataLoader(file_path)

    @staticmethod
    def create_embedding_generator(model_name='sentence-transformers/all-MiniLM-L6-v2'):
        return EmbeddingGenerator(model_name)

    @staticmethod
    def create_similarity_calculator():
        return SimilarityCalculator()

    @staticmethod
    def create_movie_search(df):
        embedding_generator = MovieComponentFactory.create_embedding_generator()
        similarity_calculator = MovieComponentFactory.create_similarity_calculator()
        return MovieSearch(df, embedding_generator, similarity_calculator)

# Uso del Factory
file_path = 'src/data/IMDBtop1000.csv'

# Crear los componentes usando la Factory
data_loader = MovieComponentFactory.create_data_loader(file_path)
df = data_loader.load_data()

# Generar embeddings
embedding_generator = MovieComponentFactory.create_embedding_generator()
df['embeddings'] = embedding_generator.generate_embeddings(df['Title']).tolist()

# Crear el sistema de búsqueda de películas
movie_search = MovieComponentFactory.create_movie_search(df)

# Ejemplo de búsqueda
results = movie_search.search_by_multiple_fields('time travel sci-fi', fields=['Title', 'Genre'])
print(results.head()['Title'])
