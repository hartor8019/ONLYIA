import os
import unittest
import pandas as pd
import HtmlTestRunner
from unittest.mock import MagicMock
from src.main_students import DataLoader, EmbeddingGenerator, SimilarityCalculator, MovieSearch

# Crear la carpeta 'test-reports' si no existe
output_dir = 'test-reports'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

   
class TestMovieSearchComponents(unittest.TestCase):

    def setUp(self):
        """Este método se ejecuta antes de cada prueba para preparar los datos necesarios."""
        # Simular un DataFrame de ejemplo
        self.df = pd.DataFrame({
            'Title': ['Inception', 'The Matrix', 'Interstellar', 'Back to the Future'],
            'Genre': ['Sci-Fi', 'Action', 'Adventure', 'Sci-Fi']
        })

        # Instancias necesarias para las pruebas
        self.data_loader = DataLoader('./data/IMDBtop1000.csv')
        self.embedding_generator = EmbeddingGenerator()
        self.similarity_calculator = SimilarityCalculator()

    def test_data_loader(self):
        """Probar que DataLoader carga correctamente un DataFrame."""
        self.data_loader.load_data = MagicMock(return_value=self.df)  # Mock para evitar cargar un archivo real
        df = self.data_loader.load_data()
        self.assertIsInstance(df, pd.DataFrame)  # Comprobar que devuelve un DataFrame
        self.assertEqual(len(df), 4)  # Comprobar que el DataFrame tiene 4 registros

    def test_embedding_generator(self):
        """Probar que EmbeddingGenerator genera embeddings correctamente."""
        # Mock del método encode para evitar la ejecución real
        self.embedding_generator.model.encode = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        embeddings = self.embedding_generator.generate_embeddings(self.df['Title'])
        
        self.assertEqual(len(embeddings), 4)  # Comprobar que generó 4 embeddings
        self.assertEqual(len(embeddings[0]), 2)  # Comprobar que los embeddings tienen tamaño 2 (mock)

    def test_similarity_calculator(self):
        """Probar que SimilarityCalculator calcula la similitud correctamente."""
        embedding1 = [0.1, 0.2]
        embedding2 = [0.1, 0.2]
        similarity = self.similarity_calculator.compute_similarity(embedding1, embedding2)
        
        self.assertAlmostEqual(similarity, 1.0, places=2)  # Similitud debe ser cercana a 1.0 (idénticos)

    def test_movie_search(self):
        """Probar la búsqueda de películas en MovieSearch."""
        # Mock para embeddings
        self.embedding_generator.generate_embeddings = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        self.df['embeddings'] = self.embedding_generator.generate_embeddings(self.df['Title'])

        # Crear la instancia de MovieSearch con los mocks
        movie_search = MovieSearch(self.df, self.embedding_generator, self.similarity_calculator)
        
        # Mock para la función compute_similarity
        self.similarity_calculator.compute_similarity = MagicMock(return_value=0.9)
        
        results = movie_search.search_by_multiple_fields('Sci-Fi', fields=['Title', 'Genre'])
        
        self.assertIsInstance(results, pd.DataFrame)  # Comprobar que retorna un DataFrame
        self.assertEqual(len(results), 4)  # Comprobar que contiene todas las filas (simuladas)
        self.assertIn('Inception', results['Title'].values)  # Comprobar que 'Inception' está en los resultados

    def test_movie_search_no_results(self):
        """Probar búsqueda sin resultados (similitud muy baja)."""
        # Mock para embeddings
        self.embedding_generator.generate_embeddings = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        self.df['embeddings'] = self.embedding_generator.generate_embeddings(self.df['Title'])

        # Crear la instancia de MovieSearch con los mocks
        movie_search = MovieSearch(self.df, self.embedding_generator, self.similarity_calculator)

        # Mock para la función compute_similarity
        self.similarity_calculator.compute_similarity = MagicMock(return_value=0.1)
        
        results = movie_search.search_by_multiple_fields('Fantasy', fields=['Title', 'Genre'])
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 4)  # Sigue conteniendo 4 filas, pero la similitud será baja

 
if __name__ == '__main__':
    unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='./test-reports'))
